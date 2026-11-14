#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <Accelerate/Accelerate.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <math.h>
#include "fishhook.h"

#if __has_include("ir_data.c")
#include "ir_data.c"
#elif __has_include("ir_data.h")
#include "ir_data.h"
#else
#error "Missing ir_data.c or ir_data.h"
#endif

#ifndef SPATIAL_MODE
#define SPATIAL_MODE 1 // 1=HeSuVi Matrix Upmix, 2=Pure Front Stereo
#endif

#define FFT_N       16384
#define FFT_HALF    8192
#define FFT_LOG2    14
#define MAX_SLOTS   32

struct SpatialUnitState {
    AudioUnit unit;
    AURenderCallback origProc;
    void *origRefCon;
    double sampleRate;
    UInt32 channels;
    bool isFloat;
    bool isNonInterleaved;
    bool valid;

    float histL[FFT_N];
    float histR[FFT_N];

    float tmpInL[FFT_HALF];
    float tmpInR[FFT_HALF];

    float xLr[FFT_HALF];
    float xLi[FFT_HALF];
    float xRr[FFT_HALF];
    float xRi[FFT_HALF];

    float yLr[FFT_HALF];
    float yLi[FFT_HALF];
    float yRr[FFT_HALF];
    float yRi[FFT_HALF];

    float accLr[FFT_HALF];
    float accLi[FFT_HALF];
    float accRr[FFT_HALF];
    float accRi[FFT_HALF];

    float timeOutL[FFT_N];
    float timeOutR[FFT_N];
};

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;
static SpatialUnitState gStates[MAX_SLOTS];
static FFTSetup gFftSetup = NULL;
static atomic_bool gHeadphonesActive = true;
static __thread int tRenderDepth = 0; // Re-entrancy guard: prevents double convolution in multi-AudioUnit graphs (e.g., Spotify)

// Precomputed IR spectra (44.1 kHz & 48 kHz)
struct IRSpectra {
    float ll_r[FFT_HALF], ll_i[FFT_HALF];
    float lr_r[FFT_HALF], lr_i[FFT_HALF];
    float rl_r[FFT_HALF], rl_i[FFT_HALF];
    float rr_r[FFT_HALF], rr_i[FFT_HALF];
};
static IRSpectra gSpec44;
static IRSpectra gSpec48;

static OSStatus (*orig_AudioUnitSetProperty)(
    AudioUnit inUnit,
    AudioUnitPropertyID inID,
    AudioUnitScope inScope,
    AudioUnitElement inElement,
    const void *inData,
    UInt32 inDataSize
) = NULL;

static OSStatus (*orig_AudioComponentInstanceDispose)(
    AudioComponentInstance inInstance
) = NULL;

static void UpdateAudioRouteState(void) {
    @autoreleasepool {
        AVAudioSession *session = [AVAudioSession sharedInstance];
        AVAudioSessionRouteDescription *route = [session currentRoute];
        NSArray<AVAudioSessionPortDescription *> *outputs = [route outputs];
        if (!outputs || [outputs count] == 0) {
            return;
        }

        bool isHeadphone = false;
        for (AVAudioSessionPortDescription *port in outputs) {
            NSString *pt = [port portType];
            NSString *name = [[port portName] lowercaseString] ?: @"";

            // Built-in phone speaker/receiver, CarPlay, AirPlay, HDMI -> always bypass
            if ([pt isEqualToString:AVAudioSessionPortBuiltInSpeaker] ||
                [pt isEqualToString:AVAudioSessionPortBuiltInReceiver] ||
                [pt isEqualToString:AVAudioSessionPortCarAudio] ||
                [pt isEqualToString:AVAudioSessionPortAirPlay] ||
                [pt isEqualToString:AVAudioSessionPortHDMI]) {
                isHeadphone = false;
                break;
            }

            // Wired headphones / USB audio
            if ([pt isEqualToString:AVAudioSessionPortHeadphones] ||
                [pt isEqualToString:AVAudioSessionPortUSBAudio]) {
                isHeadphone = true;
            }
            // Bluetooth (A2DP / LE / HFP): active unless device name explicitly indicates a speaker or car kit
            else if ([pt isEqualToString:AVAudioSessionPortBluetoothA2DP] ||
                     [pt isEqualToString:AVAudioSessionPortBluetoothLE] ||
                     [pt isEqualToString:AVAudioSessionPortBluetoothHFP]) {
                if ([name containsString:@"speaker"] ||
                    [name containsString:@"soundbar"] ||
                    [name containsString:@"homepod"] ||
                    [name containsString:@"carplay"]) {
                    isHeadphone = false;
                } else {
                    isHeadphone = true;
                }
            }
        }
        atomic_store_explicit(&gHeadphonesActive, isHeadphone, memory_order_relaxed);
    }
}

static void SetupRouteObserver(void) {
    UpdateAudioRouteState();
    [[NSNotificationCenter defaultCenter] addObserverForName:AVAudioSessionRouteChangeNotification
                                                      object:nil
                                                       queue:nil
                                                  usingBlock:^(NSNotification * _Nonnull note) {
        UpdateAudioRouteState();
    }];
}

static void PrecomputeOneIR(const float *ir, float *outR, float *outI) {
    float padded[FFT_N];
    memset(padded, 0, sizeof(padded));
    memcpy(padded, ir, SPATIAL_IR_LEN * sizeof(float));

    DSPSplitComplex split = { outR, outI };
    vDSP_ctoz((const DSPComplex *)padded, 2, &split, 1, FFT_HALF);
    vDSP_fft_zrip(gFftSetup, &split, 1, FFT_LOG2, FFT_FORWARD);
}

static void InitAllSpectra(void) {
    gFftSetup = vDSP_create_fftsetup(FFT_LOG2, kFFTRadix2);
    if (!gFftSetup) return;

#if SPATIAL_MODE == 2
    PrecomputeOneIR(IR_FRONT_44100_LL, gSpec44.ll_r, gSpec44.ll_i);
    PrecomputeOneIR(IR_FRONT_44100_LR, gSpec44.lr_r, gSpec44.lr_i);
    PrecomputeOneIR(IR_FRONT_44100_RL, gSpec44.rl_r, gSpec44.rl_i);
    PrecomputeOneIR(IR_FRONT_44100_RR, gSpec44.rr_r, gSpec44.rr_i);

    PrecomputeOneIR(IR_FRONT_48000_LL, gSpec48.ll_r, gSpec48.ll_i);
    PrecomputeOneIR(IR_FRONT_48000_LR, gSpec48.lr_r, gSpec48.lr_i);
    PrecomputeOneIR(IR_FRONT_48000_RL, gSpec48.rl_r, gSpec48.rl_i);
    PrecomputeOneIR(IR_FRONT_48000_RR, gSpec48.rr_r, gSpec48.rr_i);
#else
    PrecomputeOneIR(IR_MATRIX_44100_LL, gSpec44.ll_r, gSpec44.ll_i);
    PrecomputeOneIR(IR_MATRIX_44100_LR, gSpec44.lr_r, gSpec44.lr_i);
    PrecomputeOneIR(IR_MATRIX_44100_RL, gSpec44.rl_r, gSpec44.rl_i);
    PrecomputeOneIR(IR_MATRIX_44100_RR, gSpec44.rr_r, gSpec44.rr_i);

    PrecomputeOneIR(IR_MATRIX_48000_LL, gSpec48.ll_r, gSpec48.ll_i);
    PrecomputeOneIR(IR_MATRIX_48000_LR, gSpec48.lr_r, gSpec48.lr_i);
    PrecomputeOneIR(IR_MATRIX_48000_RL, gSpec48.rl_r, gSpec48.rl_i);
    PrecomputeOneIR(IR_MATRIX_48000_RR, gSpec48.rr_r, gSpec48.rr_i);
#endif
}

static SpatialUnitState* FindOrCreateState(AudioUnit unit) {
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (gStates[i].valid && gStates[i].unit == unit) return &gStates[i];
    }
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (!gStates[i].valid) {
            memset(&gStates[i], 0, sizeof(SpatialUnitState));
            gStates[i].unit = unit;
            gStates[i].sampleRate = 44100.0;
            gStates[i].channels = 2;
            gStates[i].isFloat = true;
            gStates[i].isNonInterleaved = true;
            gStates[i].valid = true;
            return &gStates[i];
        }
    }
    return NULL;
}

static inline void PackedComplexMul(
    const DSPSplitComplex *A,
    const float *Br,
    const float *Bi,
    DSPSplitComplex *C
) {
    DSPSplitComplex B = { (float *)Br, (float *)Bi };
    float dc = A->realp[0] * Br[0];
    float nyq = A->imagp[0] * Bi[0];
    vDSP_zvmul(A, 1, &B, 1, C, 1, FFT_HALF, 1);
    C->realp[0] = dc;
    C->imagp[0] = nyq;
}

static void ProcessFFTChunk(SpatialUnitState *st, float *inL, float *inR, UInt32 chunk) {
    if (!gFftSetup || chunk == 0 || chunk > FFT_HALF) return;

    const IRSpectra *spec = (st->sampleRate >= 46000.0) ? &gSpec48 : &gSpec44;

    memmove(st->histL, st->histL + chunk, (FFT_N - chunk) * sizeof(float));
    memcpy(st->histL + (FFT_N - chunk), inL, chunk * sizeof(float));

    memmove(st->histR, st->histR + chunk, (FFT_N - chunk) * sizeof(float));
    memcpy(st->histR + (FFT_N - chunk), inR, chunk * sizeof(float));

    DSPSplitComplex xL = { st->xLr, st->xLi };
    DSPSplitComplex xR = { st->xRr, st->xRi };
    vDSP_ctoz((const DSPComplex *)st->histL, 2, &xL, 1, FFT_HALF);
    vDSP_ctoz((const DSPComplex *)st->histR, 2, &xR, 1, FFT_HALF);
    vDSP_fft_zrip(gFftSetup, &xL, 1, FFT_LOG2, FFT_FORWARD);
    vDSP_fft_zrip(gFftSetup, &xR, 1, FFT_LOG2, FFT_FORWARD);

    DSPSplitComplex yL = { st->yLr, st->yLi };
    DSPSplitComplex yR = { st->yRr, st->yRi };
    DSPSplitComplex accL = { st->accLr, st->accLi };
    DSPSplitComplex accR = { st->accRr, st->accRi };

    PackedComplexMul(&xL, spec->ll_r, spec->ll_i, &accL);
    PackedComplexMul(&xR, spec->rl_r, spec->rl_i, &yL);
    vDSP_zvadd(&accL, 1, &yL, 1, &accL, 1, FFT_HALF);

    PackedComplexMul(&xL, spec->lr_r, spec->lr_i, &accR);
    PackedComplexMul(&xR, spec->rr_r, spec->rr_i, &yR);
    vDSP_zvadd(&accR, 1, &yR, 1, &accR, 1, FFT_HALF);

    vDSP_fft_zrip(gFftSetup, &accL, 1, FFT_LOG2, FFT_INVERSE);
    vDSP_fft_zrip(gFftSetup, &accR, 1, FFT_LOG2, FFT_INVERSE);

    vDSP_ztoc(&accL, 1, (DSPComplex *)st->timeOutL, 2, FFT_HALF);
    vDSP_ztoc(&accR, 1, (DSPComplex *)st->timeOutR, 2, FFT_HALF);

    const float scale = 0.630957f / (8.0f * (float)FFT_N); // -4.0 dB headroom
    float *validL = st->timeOutL + (FFT_N - chunk);
    float *validR = st->timeOutR + (FFT_N - chunk);
    vDSP_vsmul(validL, 1, &scale, inL, 1, chunk);
    vDSP_vsmul(validR, 1, &scale, inR, 1, chunk);

    for (UInt32 i = 0; i < chunk; i++) {
        float l = inL[i];
        float r = inR[i];
        if (l > 0.99f) l = 0.99f + 0.01f * tanhf((l - 0.99f) * 100.0f);
        else if (l < -0.99f) l = -0.99f + 0.01f * tanhf((l + 0.99f) * 100.0f);
        if (r > 0.99f) r = 0.99f + 0.01f * tanhf((r - 0.99f) * 100.0f);
        else if (r < -0.99f) r = -0.99f + 0.01f * tanhf((r + 0.99f) * 100.0f);
        inL[i] = l;
        inR[i] = r;
    }
}

static void ProcessTrueStereo(SpatialUnitState *st, float *inL, float *inR, UInt32 nFrames) {
    UInt32 offset = 0;
    while (offset < nFrames) {
        UInt32 chunk = nFrames - offset;
        if (chunk > FFT_HALF) chunk = FFT_HALF;
        ProcessFFTChunk(st, inL + offset, inR + offset, chunk);
        offset += chunk;
    }
}

static OSStatus HookedRenderCallback(
    void *inRefCon,
    AudioUnitRenderActionFlags *ioActionFlags,
    const AudioTimeStamp *inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList *ioData
) {
    SpatialUnitState *st = (SpatialUnitState *)inRefCon;
    if (!st || !st->origProc) return noErr;

    const bool isOutermost = (tRenderDepth == 0);
    tRenderDepth++;
    OSStatus err = st->origProc(st->origRefCon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData);
    tRenderDepth--;

    // Only process once at the outermost hooked AudioUnit in the render chain
    if (!isOutermost) return err;
    if (err != noErr || !ioData || inNumberFrames == 0 || !st->valid || st->channels != 2) return err;
    if (ioActionFlags && (*ioActionFlags & kAudioUnitRenderAction_OutputIsSilence)) return err;

    // Auto-Bypass when routed to phone speaker, CarPlay, AirPlay, or Bluetooth speaker
    if (!atomic_load_explicit(&gHeadphonesActive, memory_order_relaxed)) {
        return err;
    }

    if (st->isFloat) {
        if (st->isNonInterleaved && ioData->mNumberBuffers >= 2) {
            float *bufL = (float *)ioData->mBuffers[0].mData;
            float *bufR = (float *)ioData->mBuffers[1].mData;
            if (bufL && bufR && ioData->mBuffers[0].mDataByteSize >= inNumberFrames * sizeof(float)) {
                ProcessTrueStereo(st, bufL, bufR, inNumberFrames);
            }
        } else if (!st->isNonInterleaved && ioData->mNumberBuffers >= 1) {
            float *inter = (float *)ioData->mBuffers[0].mData;
            if (inter && ioData->mBuffers[0].mDataByteSize >= inNumberFrames * 2 * sizeof(float)) {
                UInt32 offset = 0;
                while (offset < inNumberFrames) {
                    UInt32 chunk = inNumberFrames - offset;
                    if (chunk > FFT_HALF) chunk = FFT_HALF;
                    for (UInt32 i = 0; i < chunk; i++) {
                        st->tmpInL[i] = inter[2 * (offset + i)];
                        st->tmpInR[i] = inter[2 * (offset + i) + 1];
                    }
                    ProcessFFTChunk(st, st->tmpInL, st->tmpInR, chunk);
                    for (UInt32 i = 0; i < chunk; i++) {
                        inter[2 * (offset + i)] = st->tmpInL[i];
                        inter[2 * (offset + i) + 1] = st->tmpInR[i];
                    }
                    offset += chunk;
                }
            }
        }
    } else if (!st->isFloat && !st->isNonInterleaved && ioData->mNumberBuffers >= 1) {
        int16_t *pcm = (int16_t *)ioData->mBuffers[0].mData;
        if (pcm && ioData->mBuffers[0].mDataByteSize >= inNumberFrames * 2 * sizeof(int16_t)) {
            const float inv = 1.0f / 32768.0f;
            UInt32 offset = 0;
            while (offset < inNumberFrames) {
                UInt32 chunk = inNumberFrames - offset;
                if (chunk > FFT_HALF) chunk = FFT_HALF;
                for (UInt32 i = 0; i < chunk; i++) {
                    st->tmpInL[i] = pcm[2 * (offset + i)] * inv;
                    st->tmpInR[i] = pcm[2 * (offset + i) + 1] * inv;
                }
                ProcessFFTChunk(st, st->tmpInL, st->tmpInR, chunk);
                for (UInt32 i = 0; i < chunk; i++) {
                    float l = st->tmpInL[i] * 32767.0f;
                    float r = st->tmpInR[i] * 32767.0f;
                    if (l > 32767.0f) l = 32767.0f; else if (l < -32768.0f) l = -32768.0f;
                    if (r > 32767.0f) r = 32767.0f; else if (r < -32768.0f) r = -32768.0f;
                    pcm[2 * (offset + i)] = (int16_t)l;
                    pcm[2 * (offset + i) + 1] = (int16_t)r;
                }
                offset += chunk;
            }
        }
    }
    return err;
}

static OSStatus hooked_AudioUnitSetProperty(
    AudioUnit inUnit,
    AudioUnitPropertyID inID,
    AudioUnitScope inScope,
    AudioUnitElement inElement,
    const void *inData,
    UInt32 inDataSize
) {
    if (inID == kAudioUnitProperty_StreamFormat && inScope == kAudioUnitScope_Input && inData && inDataSize >= sizeof(AudioStreamBasicDescription)) {
        const AudioStreamBasicDescription *asbd = (const AudioStreamBasicDescription *)inData;
        pthread_mutex_lock(&gLock);
        SpatialUnitState *st = FindOrCreateState(inUnit);
        if (st) {
            double newRate = asbd->mSampleRate > 0 ? asbd->mSampleRate : 44100.0;
            if (st->sampleRate != newRate) {
                memset(st->histL, 0, sizeof(st->histL));
                memset(st->histR, 0, sizeof(st->histR));
            }
            st->sampleRate = newRate;
            st->channels = asbd->mChannelsPerFrame;
            st->isFloat = (asbd->mFormatFlags & kAudioFormatFlagIsFloat) != 0;
            st->isNonInterleaved = (asbd->mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
        }
        pthread_mutex_unlock(&gLock);
    }

    if (inID == kAudioUnitProperty_SetRenderCallback && inScope == kAudioUnitScope_Input && inData && inDataSize >= sizeof(AURenderCallbackStruct)) {
        const AURenderCallbackStruct *cb = (const AURenderCallbackStruct *)inData;
        if (cb->inputProc && (void *)cb->inputProc != (void *)HookedRenderCallback) {
            UpdateAudioRouteState();
            AudioStreamBasicDescription asbd;
            UInt32 sz = sizeof(asbd);
            bool hasFmt = (AudioUnitGetProperty(inUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, inElement, &asbd, &sz) == noErr);

            pthread_mutex_lock(&gLock);
            SpatialUnitState *st = FindOrCreateState(inUnit);
            if (st) {
                st->origProc = cb->inputProc;
                st->origRefCon = cb->inputProcRefCon;
                if (hasFmt) {
                    st->sampleRate = asbd.mSampleRate > 0 ? asbd.mSampleRate : 44100.0;
                    st->channels = asbd.mChannelsPerFrame;
                    st->isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0;
                    st->isNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
                }
                AURenderCallbackStruct newCb;
                newCb.inputProc = (AURenderCallback)HookedRenderCallback;
                newCb.inputProcRefCon = st;
                pthread_mutex_unlock(&gLock);
                return orig_AudioUnitSetProperty(inUnit, inID, inScope, inElement, &newCb, sizeof(newCb));
            }
            pthread_mutex_unlock(&gLock);
        }
    }
    return orig_AudioUnitSetProperty(inUnit, inID, inScope, inElement, inData, inDataSize);
}

static OSStatus hooked_AudioComponentInstanceDispose(AudioComponentInstance inInstance) {
    pthread_mutex_lock(&gLock);
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (gStates[i].valid && gStates[i].unit == (AudioUnit)inInstance) {
            gStates[i].valid = false;
            gStates[i].origProc = NULL;
            gStates[i].unit = NULL;
        }
    }
    pthread_mutex_unlock(&gLock);
    return orig_AudioComponentInstanceDispose(inInstance);
}

__attribute__((constructor))
static void SpatialDSPInit(void) {
    InitAllSpectra();
    SetupRouteObserver();
    struct rebinding rebinds[] = {
        { "AudioUnitSetProperty", (void *)hooked_AudioUnitSetProperty, (void **)&orig_AudioUnitSetProperty },
        { "AudioComponentInstanceDispose", (void *)hooked_AudioComponentInstanceDispose, (void **)&orig_AudioComponentInstanceDispose }
    };
    rebind_symbols(rebinds, 2);
}
