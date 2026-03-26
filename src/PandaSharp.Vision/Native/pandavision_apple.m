/*
 * PandaVision Apple-Native Video Decoder
 * =======================================
 * Uses AVFoundation AVAssetReader for VideoToolbox hardware-accelerated decode.
 * Provides both one-shot and persistent-handle APIs.
 *
 * Build:
 *   clang -O3 -shared -o libpandavision_apple.dylib pandavision_apple.m \
 *     -framework AVFoundation -framework CoreMedia -framework CoreVideo \
 *     -framework Accelerate -framework Foundation -fobjc-arc
 */

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#include <stdlib.h>

/* ── Persistent handle for reuse across multiple operations ── */

typedef struct {
    void *asset;     /* AVAsset* (bridged) */
    void *track;     /* AVAssetTrack* (bridged) */
    int width, height;
    double fps, duration;
    int frame_count;
} AppleVideoHandle;

AppleVideoHandle* apple_video_open(const char *path)
{
    @autoreleasepool {
        NSString *filePath = [NSString stringWithUTF8String:path];
        NSURL *url = [NSURL fileURLWithPath:filePath];
        AVAsset *asset = [AVAsset assetWithURL:url];

        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        #pragma clang diagnostic pop

        if (tracks.count == 0) return NULL;
        AVAssetTrack *track = tracks[0];

        AppleVideoHandle *h = (AppleVideoHandle*)calloc(1, sizeof(AppleVideoHandle));
        h->asset = (__bridge_retained void*)asset;
        h->track = (__bridge_retained void*)track;
        CGSize size = track.naturalSize;
        h->width = (int)size.width;
        h->height = (int)size.height;
        h->fps = track.nominalFrameRate;
        h->duration = CMTimeGetSeconds(asset.duration);
        h->frame_count = (int)(h->fps * h->duration);
        return h;
    }
}

int apple_video_width(AppleVideoHandle *h) { return h ? h->width : 0; }
int apple_video_height(AppleVideoHandle *h) { return h ? h->height : 0; }
int apple_video_frame_count(AppleVideoHandle *h) { return h ? h->frame_count : 0; }
double apple_video_fps(AppleVideoHandle *h) { return h ? h->fps : 0; }
double apple_video_duration(AppleVideoHandle *h) { return h ? h->duration : 0; }

/* Extract frames using a pre-opened handle (fast — skips asset loading) */
int apple_handle_extract(AppleVideoHandle *h, float *output, int max_frames,
                          int every_nth, int target_w, int target_h)
{
    if (!h || !output) return 0;

    @autoreleasepool {
        AVAsset *asset = (__bridge AVAsset*)h->asset;
        AVAssetTrack *track = (__bridge AVAssetTrack*)h->track;
        NSError *error = nil;

        AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
        if (!reader || error) return 0;

        NSDictionary *settings = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (NSString *)kCVPixelBufferWidthKey: @(target_w),
            (NSString *)kCVPixelBufferHeightKey: @(target_h),
        };

        AVAssetReaderTrackOutput *trackOutput =
            [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:track outputSettings:settings];
        trackOutput.alwaysCopiesSampleData = NO;

        if (![reader canAddOutput:trackOutput]) return 0;
        [reader addOutput:trackOutput];
        if (![reader startReading]) return 0;

        int frame_idx = 0, extracted = 0;
        int frame_pixels = target_w * target_h * 3;
        const float s = 1.0f / 255.0f;

        CMSampleBufferRef sampleBuffer;
        while ((sampleBuffer = [trackOutput copyNextSampleBuffer]) != NULL && extracted < max_frames)
        {
            if (frame_idx % every_nth == 0)
            {
                CVImageBufferRef imgBuf = CMSampleBufferGetImageBuffer(sampleBuffer);
                if (imgBuf)
                {
                    CVPixelBufferLockBaseAddress(imgBuf, kCVPixelBufferLock_ReadOnly);
                    uint8_t *base = (uint8_t *)CVPixelBufferGetBaseAddress(imgBuf);
                    size_t bpr = CVPixelBufferGetBytesPerRow(imgBuf);

                    float *dst = output + (long)extracted * frame_pixels;
                    for (int y = 0; y < target_h; y++)
                    {
                        const uint8_t *row = base + y * bpr;
                        float *frow = dst + y * target_w * 3;
                        for (int x = 0; x < target_w; x++)
                        {
                            const uint8_t *p = row + x * 4;
                            frow[x*3]   = p[2] * s;
                            frow[x*3+1] = p[1] * s;
                            frow[x*3+2] = p[0] * s;
                        }
                    }

                    CVPixelBufferUnlockBaseAddress(imgBuf, kCVPixelBufferLock_ReadOnly);
                    extracted++;
                }
            }
            CFRelease(sampleBuffer);
            frame_idx++;
        }

        [reader cancelReading];
        return extracted;
    }
}

void apple_video_close(AppleVideoHandle *h)
{
    if (!h) return;
    if (h->asset) CFRelease(h->asset);
    if (h->track) CFRelease(h->track);
    free(h);
}

/* ── One-shot convenience (for backward compat) ── */
int apple_extract_frames(const char *path, float *output, int max_frames,
                          int every_nth, int target_w, int target_h)
{
    AppleVideoHandle *h = apple_video_open(path);
    if (!h) return 0;
    int result = apple_handle_extract(h, output, max_frames, every_nth, target_w, target_h);
    apple_video_close(h);
    return result;
}

void apple_video_info(const char *path, int *width, int *height,
                       int *frame_count, double *fps, double *duration)
{
    AppleVideoHandle *h = apple_video_open(path);
    if (!h) { *width = *height = *frame_count = 0; *fps = *duration = 0; return; }
    *width = h->width; *height = h->height;
    *frame_count = h->frame_count; *fps = h->fps; *duration = h->duration;
    apple_video_close(h);
}
