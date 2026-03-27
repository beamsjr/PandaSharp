/*
 * PandaVision Native Video Decoder
 * =================================
 * Direct FFmpeg libav* API with VideoToolbox hardware decode on macOS.
 * Falls back to multi-threaded software decode if hwaccel unavailable.
 *
 * Build (macOS, FFmpeg 7):
 *   clang -O3 -shared -o libpandavision.dylib pandavision_native.c \
 *     -I/opt/homebrew/opt/ffmpeg@7/include \
 *     -L/opt/homebrew/opt/ffmpeg@7/lib \
 *     -lavformat -lavcodec -lavutil -lswscale
 */

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    AVFormatContext *fmt_ctx;
    AVCodecContext *codec_ctx;
    int video_stream_idx;
    int width, height, frame_count;
    double fps, duration_sec;
    int has_hwaccel;
    enum AVPixelFormat hw_pix_fmt;
} VideoHandle;

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
    VideoHandle *vh = (VideoHandle*)ctx->opaque;
    for (const enum AVPixelFormat *p = pix_fmts; *p != -1; p++)
        if (*p == vh->hw_pix_fmt) return *p;
    return pix_fmts[0];
}

VideoHandle* video_open(const char* path)
{
    VideoHandle *vh = (VideoHandle*)calloc(1, sizeof(VideoHandle));
    if (!vh) return NULL;

    if (avformat_open_input(&vh->fmt_ctx, path, NULL, NULL) != 0)
    { free(vh); return NULL; }

    if (avformat_find_stream_info(vh->fmt_ctx, NULL) < 0)
    { avformat_close_input(&vh->fmt_ctx); free(vh); return NULL; }

    vh->video_stream_idx = -1;
    for (unsigned i = 0; i < vh->fmt_ctx->nb_streams; i++)
        if (vh->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        { vh->video_stream_idx = i; break; }
    if (vh->video_stream_idx < 0)
    { avformat_close_input(&vh->fmt_ctx); free(vh); return NULL; }

    AVStream *vs = vh->fmt_ctx->streams[vh->video_stream_idx];
    AVCodecParameters *codecpar = vs->codecpar;
    const AVCodec *codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec)
    { avformat_close_input(&vh->fmt_ctx); free(vh); return NULL; }

    vh->codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(vh->codec_ctx, codecpar);

    /* Multi-threaded software decode — fastest path for CPU float extraction.
     * VideoToolbox hwaccel decodes fast but GPU→CPU transfer + NV12→RGB conversion
     * is slower than threaded software decode for our use case. */
    vh->has_hwaccel = 0;
    vh->codec_ctx->thread_count = 0;
    vh->codec_ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    if (avcodec_open2(vh->codec_ctx, codec, NULL) < 0)
    {
        avcodec_free_context(&vh->codec_ctx);
        avformat_close_input(&vh->fmt_ctx);
        free(vh); return NULL;
    }

    vh->width = codecpar->width;
    vh->height = codecpar->height;
    vh->frame_count = (int)vs->nb_frames;
    if (vs->avg_frame_rate.den > 0)
        vh->fps = (double)vs->avg_frame_rate.num / vs->avg_frame_rate.den;
    if (vh->fmt_ctx->duration > 0)
        vh->duration_sec = (double)vh->fmt_ctx->duration / AV_TIME_BASE;

    return vh;
}

int video_width(VideoHandle *vh) { return vh ? vh->width : 0; }
int video_height(VideoHandle *vh) { return vh ? vh->height : 0; }
int video_frame_count(VideoHandle *vh) { return vh ? vh->frame_count : 0; }
double video_fps(VideoHandle *vh) { return vh ? vh->fps : 0; }
double video_duration(VideoHandle *vh) { return vh ? vh->duration_sec : 0; }

int video_extract_frames(VideoHandle *vh, float *output, int max_frames,
                          int every_nth, int target_w, int target_h)
{
    if (!vh || !output) return 0;

    AVFrame *frame = av_frame_alloc();
    AVFrame *sw_frame = av_frame_alloc();
    AVFrame *rgb_frame = av_frame_alloc();
    AVPacket *pkt = av_packet_alloc();

    int rgb_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, target_w, target_h, 1);
    uint8_t *rgb_buf = (uint8_t*)av_malloc(rgb_size);
    av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize,
                         rgb_buf, AV_PIX_FMT_RGB24, target_w, target_h, 1);

    struct SwsContext *sws = NULL;
    enum AVPixelFormat last_src_fmt = AV_PIX_FMT_NONE;
    int last_src_w = 0, last_src_h = 0;
    int frame_idx = 0, extracted = 0;
    int frame_pixels = target_w * target_h * 3;

    while (av_read_frame(vh->fmt_ctx, pkt) >= 0 && extracted < max_frames)
    {
        if (pkt->stream_index != vh->video_stream_idx)
        { av_packet_unref(pkt); continue; }

        if (avcodec_send_packet(vh->codec_ctx, pkt) != 0)
        { av_packet_unref(pkt); continue; }

        while (avcodec_receive_frame(vh->codec_ctx, frame) == 0)
        {
            if (frame_idx % every_nth != 0)
            { frame_idx++; continue; }

            AVFrame *src = frame;

            /* Hardware frame → transfer to CPU memory */
            if (vh->has_hwaccel && frame->format == vh->hw_pix_fmt)
            {
                /* Transfer returns NV12 or other CPU format */
                if (av_hwframe_transfer_data(sw_frame, frame, 0) < 0)
                { frame_idx++; continue; }
                sw_frame->width = frame->width;
                sw_frame->height = frame->height;
                src = sw_frame;
            }

            /* Recreate sws context when source format/size changes */
            if (src->format != last_src_fmt || src->width != last_src_w || src->height != last_src_h)
            {
                if (sws) sws_freeContext(sws);
                sws = sws_getContext(
                    src->width, src->height, src->format,
                    target_w, target_h, AV_PIX_FMT_RGB24,
                    SWS_FAST_BILINEAR, NULL, NULL, NULL);
                last_src_fmt = src->format;
                last_src_w = src->width;
                last_src_h = src->height;
            }

            if (!sws) { frame_idx++; continue; }

            /* Scale + colorspace convert to RGB24 */
            sws_scale(sws, (const uint8_t * const*)src->data,
                      src->linesize, 0, src->height,
                      rgb_frame->data, rgb_frame->linesize);

            /* RGB24 → float [0,1] */
            float *dst = output + (long)extracted * frame_pixels;
            uint8_t *rgb = rgb_frame->data[0];
            int stride = rgb_frame->linesize[0];
            for (int y = 0; y < target_h; y++)
            {
                uint8_t *row = rgb + y * stride;
                float *frow = dst + y * target_w * 3;
                for (int x = 0; x < target_w * 3; x++)
                    frow[x] = row[x] * (1.0f / 255.0f);
            }

            extracted++;
            frame_idx++;
            if (extracted >= max_frames) break;
        }
        av_packet_unref(pkt);
    }

    if (sws) sws_freeContext(sws);
    av_free(rgb_buf);
    av_frame_free(&rgb_frame);
    av_frame_free(&sw_frame);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    return extracted;
}

void video_close(VideoHandle *vh)
{
    if (!vh) return;
    if (vh->codec_ctx) avcodec_free_context(&vh->codec_ctx);
    if (vh->fmt_ctx) avformat_close_input(&vh->fmt_ctx);
    free(vh);
}
