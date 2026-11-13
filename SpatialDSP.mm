#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <Accelerate/Accelerate.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <math.h>
#include "fishhook.h"
#include "ir_data.h"

#ifndef SPATIAL_MODE
#define SPATIAL_MODE 1 // 1=HeSuVi Matrix Upmix, 2=Pure Front Stereo
#endif

#define FFT_N       8192
#define FFT_HALF    4096
#define FFT_LOG2    13
#define MAX_SLOTS   32

// vDSP_fft_zrip forward (x2) + inverse (xFFT_N) => 4*FFT_N is unity gain.
// Dividing by 8*FFT_N keeps -6.02 dB anti-clipping headroom for hot 0 dBFS tracks.
#define DSP_INV_SCALE (1.0f / (8.0f * (float)FFT_N))

struct SpatialUnitState {
    AudioUnit unit;
    AudioUnitElement element;
    AURenderCallback origProc;
    void *origRefCon;
    double sampleRate;
    UInt32 channels;
    UInt32 bitsPerChannel;
    bool isFloat;
    bool isNonInterleaved;
    bool wasBypassed;
    atomic_bool valid;

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
static id gRouteObserver = nil;

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
            // Bluetooth (A2DP / LE / HFP): active unless device name indicates a speaker or car kit
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
    gRouteObserver = [[NSNotificationCenter defaultCenter]
        addObserverForName:AVAudioSessionRouteChangeNotification
                    object:nil
                     queue:nil
                usingBlock:^(NSNotification * _Nonnull note) {
        (void)note;
        UpdateAudioRouteState();
    }];
}

static void PrecomputeOneIR(const float *ir, float *outR, float *outI) {
    float padded[FFT_N];
    memset(padded, 0, sizeof(padded));
    memcpy(padded, ir, sizeof(float) * SPATIAL_IR_LEN);

    DSPSplitComplex spec = { outR, outI };
    vDSP_ctoz((const DSPComplex *)padded, 2, &spec, 1, FFT_HALF);
    vDSP_fft_zrip(gFftSetup, &spec, 1, FFT_LOG2, FFT_FORWARD);
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

static inline void ComplexMulAccum(
    const float *xr, const float *xi,
    const float *hr, const float *hi,
    float *yr, float *yi,
    float *tmpr, float *tmpi,
    bool accumulate
) {
    DSPSplitComplex X = { (float *)xr, (float *)xi };
    DSPSplitComplex H = { (float *)hr, (float *)hi };

    if (!accumulate) {
        DSPSplitComplex Y = { yr, yi };
        vDSP_zvmul(&X, 1, &H, 1, &Y, 1, FFT_HALF, 1);
        // vDSP_fft_zrip packs DC in realp[0] and Nyquist in imagp[0] (both real)
        yr[0] = xr[0] * hr[0];
        yi[0] = xi[0] * hi[0];
    } else {
        DSPSplitComplex T = { tmpr, tmpi };
        DSPSplitComplex Y = { yr, yi };
        vDSP_zvmul(&X, 1, &H, 1, &T, 1, FFT_HALF, 1);
        tmpr[0] = xr[0] * hr[0];
        tmpi[0] = xi[0] * hi[0];
        vDSP_zvadd(&Y, 1, &T, 1, &Y, 1, FFT_HALF);
    }
}

static void ProcessFFTChunk(SpatialUnitState *st, float *inOutL, float *inOutR, UInt32 nFrames) {
    if (!gFftSetup || nFrames == 0 || nFrames > FFT_HALF) return;

    const IRSpectra *spec = (st->sampleRate <= 0.0 || st->sampleRate >= 46000.0) ? &gSpec48 : &gSpec44;

    // Shift history left by nFrames and append new input at the tail
    const UInt32 keep = FFT_N - nFrames;
    memmove(st->histL, st->histL + nFrames, keep * sizeof(float));
    memmove(st->histR, st->histR + nFrames, keep * sizeof(float));
    memcpy(st->histL + keep, inOutL, nFrames * sizeof(float));
    memcpy(st->histR + keep, inOutR, nFrames * sizeof(float));

    // Forward real-to-complex FFT
    DSPSplitComplex XL = { st->xLr, st->xLi };
    DSPSplitComplex XR = { st->xRr, st->xRi };
    vDSP_ctoz((const DSPComplex *)st->histL, 2, &XL, 1, FFT_HALF);
    vDSP_ctoz((const DSPComplex *)st->histR, 2, &XR, 1, FFT_HALF);
    vDSP_fft_zrip(gFftSetup, &XL, 1, FFT_LOG2, FFT_FORWARD);
    vDSP_fft_zrip(gFftSetup, &XR, 1, FFT_LOG2, FFT_FORWARD);

    // True-stereo matrix:
    // OutL = InL * H_LL + InR * H_RL
    // OutR = InL * H_LR + InR * H_RR
    ComplexMulAccum(st->xLr, st->xLi, spec->ll_r, spec->ll_i, st->yLr, st->yLi, st->accLr, st->accLi, false);
    ComplexMulAccum(st->xRr, st->xRi, spec->rl_r, spec->rl_i, st->yLr, st->yLi, st->accLr, st->accLi, true);

    ComplexMulAccum(st->xLr, st->xLi, spec->lr_r, spec->lr_i, st->yRr, st->yRi, st->accRr, st->accRi, false);
    ComplexMulAccum(st->xRr, st->xRi, spec->rr_r, spec->rr_i, st->yRr, st->yRi, st->accRr, st->accRi, true);

    // Inverse FFT
    DSPSplitComplex YL = { st->yLr, st->yLi };
    DSPSplitComplex YR = { st->yRr, st->yRi };
    vDSP_fft_zrip(gFftSetup, &YL, 1, FFT_LOG2, FFT_INVERSE);
    vDSP_fft_zrip(gFftSetup, &YR, 1, FFT_LOG2, FFT_INVERSE);

    vDSP_ztoc(&YL, 1, (DSPComplex *)st->timeOutL, 2, FFT_HALF);
    vDSP_ztoc(&YR, 1, (DSPComplex *)st->timeOutR, 2, FFT_HALF);

    // Overlap-Save: extract the non-aliased tail [FFT_N - nFrames .. FFT_N - 1]
    const float scale = DSP_INV_SCALE;
    vDSP_vsmul(st->timeOutL + keep, 1, &scale, inOutL, 1, nFrames);
    vDSP_vsmul(st->timeOutR + keep, 1, &scale, inOutR, 1, nFrames);
}

static inline void ResetStateHistory(SpatialUnitState *st) {
    memset(st->histL, 0, sizeof(st->histL));
    memset(st->histR, 0, sizeof(st->histR));
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
    if (!st || !atomic_load_explicit(&st->valid, memory_order_acquire) || !st->origProc) {
        return noErr;
    }

    OSStatus err = st->origProc(st->origRefCon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData);
    if (err != noErr || !ioData || inNumberFrames == 0) {
        return err;
    }

    const bool isSilence = (ioActionFlags && (*ioActionFlags & kAudioUnitRenderAction_OutputIsSilence));
    const bool active = atomic_load_explicit(&gHeadphonesActive, memory_order_relaxed);
    if (!active || isSilence) {
        if (!st->wasBypassed) {
            ResetStateHistory(st);
            st->wasBypassed = true;
        }
        return err;
    }
    st->wasBypassed = false;

    if (st->isFloat) {
        if (st->isNonInterleaved) {
            if (ioData->mNumberBuffers < 2) return err;
            float *bufL = (float *)ioData->mBuffers[0].mData;
            float *bufR = (float *)ioData->mBuffers[1].mData;
            if (!bufL || !bufR) return err;

            UInt32 availL = ioData->mBuffers[0].mDataByteSize / sizeof(float);
            UInt32 availR = ioData->mBuffers[1].mDataByteSize / sizeof(float);
            UInt32 total = inNumberFrames;
            if (availL < total) total = availL;
            if (availR < total) total = availR;

            UInt32 offset = 0;
            while (offset < total) {
                UInt32 chunk = total - offset;
                if (chunk > FFT_HALF) chunk = FFT_HALF;
                ProcessFFTChunk(st, bufL + offset, bufR + offset, chunk);
                offset += chunk;
            }
        } else {
            if (ioData->mNumberBuffers < 1 || st->channels < 2) return err;
            float *buf = (float *)ioData->mBuffers[0].mData;
            if (!buf) return err;

            UInt32 avail = ioData->mBuffers[0].mDataByteSize / (sizeof(float) * st->channels);
            UInt32 total = (inNumberFrames < avail) ? inNumberFrames : avail;
            UInt32 ch = st->channels;

            UInt32 offset = 0;
            while (offset < total) {
                UInt32 chunk = total - offset;
                if (chunk > FFT_HALF) chunk = FFT_HALF;
                for (UInt32 i = 0; i < chunk; i++) {
                    st->tmpInL[i] = buf[(offset + i) * ch];
                    st->tmpInR[i] = buf[(offset + i) * ch + 1];
                }
                ProcessFFTChunk(st, st->tmpInL, st->tmpInR, chunk);
                for (UInt32 i = 0; i < chunk; i++) {
                    buf[(offset + i) * ch]     = st->tmpInL[i];
                    buf[(offset + i) * ch + 1] = st->tmpInR[i];
                }
                offset += chunk;
            }
        }
    } else if (!st->isNonInterleaved && st->bitsPerChannel == 16 && st->channels >= 2) {
        if (ioData->mNumberBuffers < 1) return err;
        int16_t *buf = (int16_t *)ioData->mBuffers[0].mData;
        if (!buf) return err;

        UInt32 avail = ioData->mBuffers[0].mDataByteSize / (sizeof(int16_t) * st->channels);
        UInt32 total = (inNumberFrames < avail) ? inNumberFrames : avail;
        UInt32 ch = st->channels;
        const float toFloat = 1.0f / 32768.0f;

        UInt32 offset = 0;
        while (offset < total) {
            UInt32 chunk = total - offset;
            if (chunk > FFT_HALF) chunk = FFT_HALF;
            for (UInt32 i = 0; i < chunk; i++) {
                st->tmpInL[i] = (float)buf[(offset + i) * ch] * toFloat;
                st->tmpInR[i] = (float)buf[(offset + i) * ch + 1] * toFloat;
            }
            ProcessFFTChunk(st, st->tmpInL, st->tmpInR, chunk);
            for (UInt32 i = 0; i < chunk; i++) {
                float l = st->tmpInL[i] * 32767.0f;
                float r = st->tmpInR[i] * 32767.0f;
                if (l > 32767.0f) l = 32767.0f; else if (l < -32768.0f) l = -32768.0f;
                if (r > 32767.0f) r = 32767.0f; else if (r < -32768.0f) r = -32768.0f;
                buf[(offset + i) * ch]     = (int16_t)l;
                buf[(offset + i) * ch + 1] = (int16_t)r;
            }
            offset += chunk;
        }
    }

    return err;
}

static SpatialUnitState *FindOrCreateState(AudioUnit unit, AudioUnitElement element) {
    SpatialUnitState *freeSlot = NULL;
    for (int i = 0; i < MAX_SLOTS; i++) {
        bool v = atomic_load_explicit(&gStates[i].valid, memory_order_relaxed);
        if (v && gStates[i].unit == unit && gStates[i].element == element) {
            return &gStates[i];
        }
        if (!v && !freeSlot) {
            freeSlot = &gStates[i];
        }
    }
    if (freeSlot) {
        memset(freeSlot, 0, sizeof(SpatialUnitState));
        freeSlot->unit = unit;
        freeSlot->element = element;
        freeSlot->sampleRate = 48000.0;
        freeSlot->channels = 2;
        freeSlot->bitsPerChannel = 32;
        freeSlot->isFloat = true;
        freeSlot->isNonInterleaved = true;
        freeSlot->wasBypassed = false;
        atomic_store_explicit(&freeSlot->valid, true, memory_order_release);
    }
    return freeSlot;
}

static void RefreshStreamFormat(SpatialUnitState *st, AudioUnit unit, AudioUnitElement element, const AudioStreamBasicDescription *asbdOpt) {
    AudioStreamBasicDescription asbd;
    if (asbdOpt) {
        asbd = *asbdOpt;
    } else {
        UInt32 size = sizeof(asbd);
        if (AudioUnitGetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, element, &asbd, &size) != noErr) {
            return;
        }
    }

    bool changed = (st->sampleRate != asbd.mSampleRate || st->channels != asbd.mChannelsPerFrame);
    st->sampleRate       = asbd.mSampleRate;
    st->channels         = asbd.mChannelsPerFrame;
    st->bitsPerChannel   = asbd.mBitsPerChannel;
    st->isFloat          = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0;
    st->isNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
    if (changed) {
        ResetStateHistory(st);
    }
}

static OSStatus hooked_AudioUnitSetProperty(
    AudioUnit inUnit,
    AudioUnitPropertyID inID,
    AudioUnitScope inScope,
    AudioUnitElement inElement,
    const void *inData,
    UInt32 inDataSize
) {
    if (inID == kAudioUnitProperty_StreamFormat &&
        inData && inDataSize >= sizeof(AudioStreamBasicDescription)) {
        pthread_mutex_lock(&gLock);
        for (int i = 0; i < MAX_SLOTS; i++) {
            if (atomic_load_explicit(&gStates[i].valid, memory_order_relaxed) &&
                gStates[i].unit == inUnit && gStates[i].element == inElement) {
                RefreshStreamFormat(&gStates[i], inUnit, inElement, (const AudioStreamBasicDescription *)inData);
            }
        }
        pthread_mutex_unlock(&gLock);
    } else if ((inID == kAudioUnitProperty_SetRenderCallback || inID == kAudioOutputUnitProperty_SetInputCallback) &&
               inData && inDataSize >= sizeof(AURenderCallbackStruct)) {
        const AURenderCallbackStruct *cb = (const AURenderCallbackStruct *)inData;

        pthread_mutex_lock(&gLock);
        if (cb->inputProc == NULL) {
            for (int i = 0; i < MAX_SLOTS; i++) {
                if (atomic_load_explicit(&gStates[i].valid, memory_order_relaxed) &&
                    gStates[i].unit == inUnit && gStates[i].element == inElement) {
                    atomic_store_explicit(&gStates[i].valid, false, memory_order_release);
                }
            }
            pthread_mutex_unlock(&gLock);
            return orig_AudioUnitSetProperty(inUnit, inID, inScope, inElement, inData, inDataSize);
        }

        if ((const void *)cb->inputProc != (const void *)HookedRenderCallback) {
            SpatialUnitState *st = FindOrCreateState(inUnit, inElement);
            if (st) {
                st->origProc   = cb->inputProc;
                st->origRefCon = cb->inputProcRefCon;
                RefreshStreamFormat(st, inUnit, inElement, NULL);

                AURenderCallbackStruct replacement;
                replacement.inputProc       = (AURenderCallback)HookedRenderCallback;
                replacement.inputProcRefCon = st;

                pthread_mutex_unlock(&gLock);
                return orig_AudioUnitSetProperty(inUnit, inID, inScope, inElement, &replacement, sizeof(replacement));
            }
        }
        pthread_mutex_unlock(&gLock);
    }

    return orig_AudioUnitSetProperty(inUnit, inID, inScope, inElement, inData, inDataSize);
}

static OSStatus hooked_AudioComponentInstanceDispose(AudioComponentInstance inInstance) {
    pthread_mutex_lock(&gLock);
    for (int i = 0; i < MAX_SLOTS; i++) {
        if (atomic_load_explicit(&gStates[i].valid, memory_order_relaxed) &&
            gStates[i].unit == (AudioUnit)inInstance) {
            atomic_store_explicit(&gStates[i].valid, false, memory_order_release);
        }
    }
    pthread_mutex_unlock(&gLock);
    return orig_AudioComponentInstanceDispose(inInstance);
}

__attribute__((constructor))
static void SpatialDSPInit(void) {
    memset(gStates, 0, sizeof(gStates));
    InitAllSpectra();
    SetupRouteObserver();

    struct rebinding rebindings[] = {
        { "AudioUnitSetProperty", (void *)hooked_AudioUnitSetProperty, (void **)&orig_AudioUnitSetProperty },
        { "AudioComponentInstanceDispose", (void *)hooked_AudioComponentInstanceDispose, (void **)&orig_AudioComponentInstanceDispose }
    };
    rebind_symbols(rebindings, sizeof(rebindings) / sizeof(rebindings[0]));
}
