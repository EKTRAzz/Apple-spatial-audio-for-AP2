#import <Foundation/Foundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <Accelerate/Accelerate.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#include "fishhook.h"
#include "ir_data.h"

#ifndef SPATIAL_MODE
#define SPATIAL_MODE 1 // 1=HeSuVi Matrix Upmix, 2=Pure Front Stereo
#endif

#define MAX_FRAMES 4096

struct SpatialUnitState {
    AudioUnit unit;
    AURenderCallback origProc;
    void *origRefCon;
    double sampleRate;
    UInt32 channels;
    bool isFloat;
    bool isNonInterleaved;
    bool valid;

    float histL[SPATIAL_IR_LEN + MAX_FRAMES];
    float histR[SPATIAL_IR_LEN + MAX_FRAMES];
    float outLL[MAX_FRAMES];
    float outLR[MAX_FRAMES];
    float outRL[MAX_FRAMES];
    float outRR[MAX_FRAMES];
    float tmpL[MAX_FRAMES];
    float tmpR[MAX_FRAMES];
};

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;
static SpatialUnitState gStates[16];
static OSStatus (*orig_AudioUnitSetProperty)(
    AudioUnit inUnit,
    AudioUnitPropertyID inID,
    AudioUnitScope inScope,
    AudioUnitElement inElement,
    const void *inData,
    UInt32 inDataSize
) = NULL;

static SpatialUnitState* FindOrCreateState(AudioUnit unit) {
    for (int i = 0; i < 16; i++) {
        if (gStates[i].valid && gStates[i].unit == unit) return &gStates[i];
    }
    for (int i = 0; i < 16; i++) {
        if (!gStates[i].valid) {
            memset(&gStates[i], 0, sizeof(SpatialUnitState));
            gStates[i].unit = unit;
            gStates[i].sampleRate = 48000.0;
            gStates[i].channels = 2;
            gStates[i].isFloat = true;
            gStates[i].isNonInterleaved = true;
            gStates[i].valid = true;
            return &gStates[i];
        }
    }
    return NULL;
}

static void SelectTrueStereoIR(double sampleRate, const float **ll, const float **lr, const float **rl, const float **rr) {
    bool is48 = (sampleRate >= 46000.0);
#if SPATIAL_MODE == 2
    *ll = is48 ? IR_FRONT_48000_LL : IR_FRONT_44100_LL;
    *lr = is48 ? IR_FRONT_48000_LR : IR_FRONT_44100_LR;
    *rl = is48 ? IR_FRONT_48000_RL : IR_FRONT_44100_RL;
    *rr = is48 ? IR_FRONT_48000_RR : IR_FRONT_44100_RR;
#else
    *ll = is48 ? IR_MATRIX_48000_LL : IR_MATRIX_44100_LL;
    *lr = is48 ? IR_MATRIX_48000_LR : IR_MATRIX_44100_LR;
    *rl = is48 ? IR_MATRIX_48000_RL : IR_MATRIX_44100_RL;
    *rr = is48 ? IR_MATRIX_48000_RR : IR_MATRIX_44100_RR;
#endif
}

static void ProcessTrueStereo(SpatialUnitState *st, float *inL, float *inR, UInt32 nFrames) {
    if (nFrames == 0 || nFrames > MAX_FRAMES) return;

    const float *ll, *lr, *rl, *rr;
    SelectTrueStereoIR(st->sampleRate, &ll, &lr, &rl, &rr);

    memmove(st->histL, st->histL + nFrames, (SPATIAL_IR_LEN - 1) * sizeof(float));
    memcpy(st->histL + (SPATIAL_IR_LEN - 1), inL, nFrames * sizeof(float));

    memmove(st->histR, st->histR + nFrames, (SPATIAL_IR_LEN - 1) * sizeof(float));
    memcpy(st->histR + (SPATIAL_IR_LEN - 1), inR, nFrames * sizeof(float));

    // True-Stereo 4-path convolution: L_out = L*LL + R*RL, R_out = L*LR + R*RR
    vDSP_conv(st->histL, 1, ll + (SPATIAL_IR_LEN - 1), -1, st->outLL, 1, nFrames, SPATIAL_IR_LEN);
    vDSP_conv(st->histR, 1, rl + (SPATIAL_IR_LEN - 1), -1, st->outRL, 1, nFrames, SPATIAL_IR_LEN);
    vDSP_conv(st->histL, 1, lr + (SPATIAL_IR_LEN - 1), -1, st->outLR, 1, nFrames, SPATIAL_IR_LEN);
    vDSP_conv(st->histR, 1, rr + (SPATIAL_IR_LEN - 1), -1, st->outRR, 1, nFrames, SPATIAL_IR_LEN);

    vDSP_vadd(st->outLL, 1, st->outRL, 1, inL, 1, nFrames);
    vDSP_vadd(st->outLR, 1, st->outRR, 1, inR, 1, nFrames);

    for (UInt32 i = 0; i < nFrames; i++) {
        float l = inL[i];
        float r = inR[i];
        if (l > 0.98f) l = 0.98f + 0.02f * tanhf((l - 0.98f) * 50.0f);
        else if (l < -0.98f) l = -0.98f + 0.02f * tanhf((l + 0.98f) * 50.0f);
        if (r > 0.98f) r = 0.98f + 0.02f * tanhf((r - 0.98f) * 50.0f);
        else if (r < -0.98f) r = -0.98f + 0.02f * tanhf((r + 0.98f) * 50.0f);
        inL[i] = l;
        inR[i] = r;
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

    OSStatus err = st->origProc(st->origRefCon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData);
    if (err != noErr || !ioData || inNumberFrames == 0 || inNumberFrames > MAX_FRAMES) return err;
    if (ioActionFlags && (*ioActionFlags & kAudioUnitRenderAction_OutputIsSilence)) return err;

    if (st->isFloat) {
        if (st->isNonInterleaved && ioData->mNumberBuffers >= 2) {
            float *bufL = (float *)ioData->mBuffers[0].mData;
            float *bufR = (float *)ioData->mBuffers[1].mData;
            if (bufL && bufR) ProcessTrueStereo(st, bufL, bufR, inNumberFrames);
        } else if (!st->isNonInterleaved && ioData->mNumberBuffers >= 1 && st->channels == 2) {
            float *inter = (float *)ioData->mBuffers[0].mData;
            if (inter) {
                for (UInt32 i = 0; i < inNumberFrames; i++) {
                    st->tmpL[i] = inter[2 * i];
                    st->tmpR[i] = inter[2 * i + 1];
                }
                ProcessTrueStereo(st, st->tmpL, st->tmpR, inNumberFrames);
                for (UInt32 i = 0; i < inNumberFrames; i++) {
                    inter[2 * i] = st->tmpL[i];
                    inter[2 * i + 1] = st->tmpR[i];
                }
            }
        }
    } else if (!st->isFloat && !st->isNonInterleaved && ioData->mNumberBuffers >= 1 && st->channels == 2) {
        int16_t *pcm = (int16_t *)ioData->mBuffers[0].mData;
        if (pcm) {
            const float inv = 1.0f / 32768.0f;
            for (UInt32 i = 0; i < inNumberFrames; i++) {
                st->tmpL[i] = pcm[2 * i] * inv;
                st->tmpR[i] = pcm[2 * i + 1] * inv;
            }
            ProcessTrueStereo(st, st->tmpL, st->tmpR, inNumberFrames);
            for (UInt32 i = 0; i < inNumberFrames; i++) {
                float l = st->tmpL[i] * 32767.0f;
                float r = st->tmpR[i] * 32767.0f;
                if (l > 32767.0f) l = 32767.0f; else if (l < -32768.0f) l = -32768.0f;
                if (r > 32767.0f) r = 32767.0f; else if (r < -32768.0f) r = -32768.0f;
                pcm[2 * i] = (int16_t)l;
                pcm[2 * i + 1] = (int16_t)r;
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
        if (st && asbd->mChannelsPerFrame == 2) {
            st->sampleRate = asbd->mSampleRate > 0 ? asbd->mSampleRate : 48000.0;
            st->channels = asbd->mChannelsPerFrame;
            st->isFloat = (asbd->mFormatFlags & kAudioFormatFlagIsFloat) != 0;
            st->isNonInterleaved = (asbd->mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
        }
        pthread_mutex_unlock(&gLock);
    }

    if (inID == kAudioUnitProperty_SetRenderCallback && inScope == kAudioUnitScope_Input && inData && inDataSize >= sizeof(AURenderCallbackStruct)) {
        const AURenderCallbackStruct *cb = (const AURenderCallbackStruct *)inData;
        if (cb->inputProc && cb->inputProc != HookedRenderCallback) {
            pthread_mutex_lock(&gLock);
            SpatialUnitState *st = FindOrCreateState(inUnit);
            if (st) {
                st->origProc = cb->inputProc;
                st->origRefCon = cb->inputProcRefCon;
                AudioStreamBasicDescription asbd;
                UInt32 sz = sizeof(asbd);
                if (AudioUnitGetProperty(inUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, inElement, &asbd, &sz) == noErr) {
                    if (asbd.mChannelsPerFrame == 2) {
                        st->sampleRate = asbd.mSampleRate > 0 ? asbd.mSampleRate : 48000.0;
                        st->channels = asbd.mChannelsPerFrame;
                        st->isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0;
                        st->isNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;
                    }
                }
                AURenderCallbackStruct newCb;
                newCb.inputProc = HookedRenderCallback;
                newCb.inputProcRefCon = st;
                pthread_mutex_unlock(&gLock);
                return orig_AudioUnitSetProperty(inUnit, inID, inScope, inElement, &newCb, sizeof(newCb));
            }
            pthread_mutex_unlock(&gLock);
        }
    }
    return orig_AudioUnitSetProperty(inUnit, inID, inScope, inElement, inData, inDataSize);
}

__attribute__((constructor))
static void SpatialDSPInit(void) {
    struct rebinding rebinds[] = {
        { "AudioUnitSetProperty", (void *)hooked_AudioUnitSetProperty, (void **)&orig_AudioUnitSetProperty }
    };
    rebind_symbols(rebinds, 1);
}
