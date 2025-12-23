# -*- encoding: utf-8 -*-
"""
FunASR WebSocket Server (流式语音识别服务)
========================================
功能: 提供基于 WebSocket 的实时语音识别服务。
支持:
  - 实时 VAD (语音活动检测)
  - 流式 ASR (在线实时识别)
  - 离线/2pass ASR (句子结束后的高精度修正)
  - 标点恢复 (Punctuation Restoration)
  - 多用户并发 (基于 ThreadPoolExecutor)

作者: 凌封 aibook.ren(AI全书)
日期: 2025-12
"""
"""
FunASR-Nano-2512 WebSocket Server
作者：凌封
来源：https://aibook.ren (AI全书)
"""
import asyncio
import json
import websockets
import time
import logging
import argparse
import ssl
import os
import numpy as np
import torch
import traceback
from concurrent.futures import ThreadPoolExecutor
# 需要引下这个，不然会报错AssertionError: FunASRNano is not registered
# issue见：https://github.com/modelscope/FunASR/issues/2741
from funasr.models.fun_asr_nano.model import FunASRNano
from funasr import AutoModel

logging.basicConfig(level=logging.ERROR)

# 全局线程池，用于并发执行模型推理，避免阻塞 asyncio 事件循环
# 建议设置为预期的最大并发数，例如 10
inference_executor = ThreadPoolExecutor(max_workers=10)

def get_args():
    """
    解析命令行参数。
    包含了服务端监听配置、模型路径配置、硬件设备配置等。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", required=False, help="监听 IP，localhost 或 0.0.0.0"
    )
    parser.add_argument("--port", type=int, default=10095, required=False, help="服务端口")
    parser.add_argument(
        "--asr_model",
        type=str,
        default="FunAudioLLM/Fun-ASR-Nano-2512",
        help="离线 ASR 模型名称 (从 ModelScope 下载)",
    )
    parser.add_argument("--asr_model_revision", type=str, default=None, help="模型版本")
    parser.add_argument(
        "--asr_model_online",
        type=str,
        default="FunAudioLLM/Fun-ASR-Nano-2512",
        help="流式 ASR 模型名称 (从 ModelScope 下载)",
    )
    parser.add_argument("--asr_model_online_revision", type=str, default=None, help="模型版本")
    parser.add_argument(
        "--vad_model",
        type=str,
        default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        help="VAD 模型名称",
    )
    parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="模型版本")
    parser.add_argument(
        "--punc_model",
        type=str,
        default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        help="标点恢复模型名称",
    )
    parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="模型版本")
    parser.add_argument("--ngpu", type=int, default=1, help="GPU 数量，0 为 CPU")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备: cuda 或 cpu")
    parser.add_argument("--ncpu", type=int, default=4, help="CPU 核心数")
    parser.add_argument(
        "--certfile",
        type=str,
        default="",
        required=False,
        help="SSL 证书文件",
    )
    parser.add_argument(
        "--keyfile",
        type=str,
        default="",
        required=False,
        help="SSL 密钥文件",
    )
    parser.add_argument("--fp16", action="store_true", help="使用 fp16 进行推理")
    return parser.parse_args()

args = get_args()

websocket_users = set()

print("正在加载模型...", flush=True)

# ASR 模型 (离线/2pass + 在线/流式)
# [优化] 合并模型加载：
# 原逻辑分别加载了离线和在线模型，导致显存双倍占用 (~7GB)。
# 经过分析，FunASR 的 AutoModel 是无状态 (Stateless) 的，状态由 status_dict 外部传入。
# 因此，我们可以共用同一个模型实例，同时服务离线和流式请求，预计节省 2-3GB 显存。
print(f"正在加载 ASR 模型: {args.asr_model} ...", flush=True)
model_asr = AutoModel(
    model=args.asr_model,
    model_revision=args.asr_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
    fp16=args.fp16,
)

# 共享同一个实例
model_asr_streaming = model_asr

# VAD 模型
model_vad = AutoModel(
    model=args.vad_model,
    model_revision=args.vad_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
    fp16=args.fp16,
)

# 标点模型
if args.punc_model != "":
    model_punc = AutoModel(
        model=args.punc_model,
        model_revision=args.punc_model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
        fp16=args.fp16,
    )
else:
    model_punc = None

print("模型加载完成！目前支持简单的多用户并发。", flush=True)

# 异步模型推理辅助函数
async def run_model_inference(model, input, **kwargs):
    """
    在线程池中运行模型推理，避免阻塞 asyncio 主事件循环。
    
    Args:
        model: FunASR 模型实例
        input: 模型输入数据 (通常是 Tensor 列表或文本)
        **kwargs: 额外的推理参数 (如 status_dict)
    
    Returns:
        推理结果
    """
    loop = asyncio.get_running_loop()
    # 使用线程池执行同步的 blocking generate 方法
    return await loop.run_in_executor(
        inference_executor, 
        lambda: model.generate(input=input, **kwargs)
    )

def decode_audio_chunk(chunk_bytes):
    """
    将接收到的原始音频字节流解码为 PyTorch Tensor。
    输入格式默认假设为: PCM, 16000Hz, 16bit, Mono.
    
    Args:
        chunk_bytes (bytes): 原始音频二进制数据
    
    Returns:
        torch.Tensor:float32: 归一化到 [-1.0, 1.0] 的音频张量
    """
    # 1. Bytes -> Int16 Numpy
    data_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
    # 2. Int16 -> Float32 (Normalize to -1.0 ~ 1.0)
    data_float32 = data_int16.astype(np.float32) / 32768.0
    # 3. Numpy -> Torch Tensor
    return torch.from_numpy(data_float32)


async def ws_reset(websocket):
    """
    重置 WebSocket 连接对应的状态缓存。
    当连接断开及为了安全起见清理内存时调用。
    """
    print("ws reset now, total num is ", len(websocket_users))
    if hasattr(websocket, "status_dict_asr_online"):
        websocket.status_dict_asr_online["cache"] = {}
        websocket.status_dict_asr_online["is_final"] = True
    if hasattr(websocket, "status_dict_vad"):
        websocket.status_dict_vad["cache"] = {}
        websocket.status_dict_vad["is_final"] = True
    if hasattr(websocket, "status_dict_punc"):
        websocket.status_dict_punc["cache"] = {}
    
    await websocket.close()


async def clear_websocket():
    """
    清理所有活跃的 WebSocket 连接。
    """
    for websocket in websocket_users:
        await ws_reset(websocket)
    websocket_users.clear()


async def ws_serve(websocket, path=None):
    """
    WebSocket 服务端主处理逻辑。
    负责处理单个客户端的完整生命周期：握手 -> 音频流处理 -> 返回结果 -> 断开。
    """
    frames = [] 
    frames_asr = [] # 离线 ASR 缓冲区 (由 VAD 分割)
    frames_asr_online = [] # 在线流式 ASR 缓冲区
    
    
    global websocket_users
    websocket_users.add(websocket)
    
    # 初始化状态字典 (参考官方示例)
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}
    websocket.status_dict_punc = {"cache": {}}
    
    websocket.chunk_interval = 10
    websocket.vad_pre_idx = 0
    speech_start = False
    speech_end_i = -1
    websocket.wav_name = "microphone"
    websocket.mode = "2pass"
    
    print("new user connected", flush=True)

    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    messagejson = json.loads(message)
                    
                    if "is_speaking" in messagejson:
                        websocket.is_speaking = messagejson["is_speaking"]
                        websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
                    if "chunk_interval" in messagejson:
                        websocket.chunk_interval = messagejson["chunk_interval"]
                    if "wav_name" in messagejson:
                        websocket.wav_name = messagejson.get("wav_name")
                    if "chunk_size" in messagejson:
                        chunk_size = messagejson["chunk_size"]
                        if isinstance(chunk_size, str):
                            chunk_size = chunk_size.split(",")
                        websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]
                    if "encoder_chunk_look_back" in messagejson:
                        websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson["encoder_chunk_look_back"]
                    if "decoder_chunk_look_back" in messagejson:
                        websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson["decoder_chunk_look_back"]
                    if "hotwords" in messagejson:
                        websocket.status_dict_asr["hotword"] = messagejson["hotwords"]
                    if "mode" in messagejson:
                        websocket.mode = messagejson["mode"]
                        # 兼容 Java 客户端:
                        # Java 客户端请求 "online" 模式，但代码逻辑依赖 "is_final": True 才能结束等待。
                        # 而 "is_final": True 是由离线识别 (async_asr) 步骤产生的。
                        # 因此，这里强制将模式升级为 "2pass" (2pass = online流式 + offline离线修正)，确保流程完整。
                        if websocket.mode == "online":
                            websocket.mode = "2pass"
                except Exception as e:
                    print("JSON error:", e)

            # 确保 VAD 的分块大小正确计算
            if "chunk_size" in websocket.status_dict_asr_online:
                 websocket.status_dict_vad["chunk_size"] = int(
                    websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval
                )
            
            # 处理音频数据
            if len(frames_asr_online) > 0 or len(frames_asr) >= 0 or not isinstance(message, str):
                if not isinstance(message, str):
                    # 收到的是音频块
                    frames.append(message)
                    duration_ms = len(message) // 32 # 16k rate, 16bit = 2 bytes. 1ms = 16 samples = 32 bytes
                    websocket.vad_pre_idx += duration_ms

                    # 1. 送入在线流式 ASR (Online ASR)
                    frames_asr_online.append(message)
                    websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
                    
                    # 根据 chunk 间隔或语音结束信号触发在线推理
                    if (len(frames_asr_online) % websocket.chunk_interval == 0 
                        or websocket.status_dict_asr_online["is_final"]):
                        
                        if websocket.mode == "2pass" or websocket.mode == "online":
                            audio_in = b"".join(frames_asr_online)
                            try:
                                await async_asr_online(websocket, audio_in)
                            except Exception as e:
                                print(f"error in asr streaming: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        frames_asr_online = [] # 清空在线缓冲区

                    # 2. 送入 VAD 检测
                    if speech_start:
                        frames_asr.append(message) # 收集用于离线识别的音频
                    
                    try:
                        speech_start_i, speech_end_i = await async_vad(websocket, message)
                    except Exception as e:
                        print("error in vad", e)
                        speech_start_i, speech_end_i = -1, -1
                    
                    # 处理 VAD 的语音开始信号
                    if speech_start_i != -1:
                        speech_start = True
                        # 回溯音频池，捕获语音起始段
                        beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
                        frames_pre = frames[-beg_bias:]
                        frames_asr = []
                        frames_asr.extend(frames_pre)
                
                # 3. 处理语音结束或流结束 -> 触发离线 ASR + 标点恢复
                if speech_end_i != -1 or not websocket.is_speaking:
                    if websocket.mode == "2pass" or websocket.mode == "offline":
                        audio_in = b"".join(frames_asr)
                        try:
                            await async_asr(websocket, audio_in)
                        except Exception as e:
                            print(f"error in asr offline: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 重置状态
                    frames_asr = []
                    speech_start = False
                    frames_asr_online = []
                    websocket.status_dict_asr_online["cache"] = {}
                    
                    if not websocket.is_speaking:
                        websocket.vad_pre_idx = 0
                        frames = []
                        websocket.status_dict_vad["cache"] = {}
                    else:
                        # 保留少量上下文
                        frames = frames[-20:]

    except websockets.ConnectionClosed:
        print("连接已关闭。", websocket_users, flush=True)
        await ws_reset(websocket)
        if websocket in websocket_users:
            websocket_users.remove(websocket)
    except Exception as e:
        print("Exception:", e)
        import traceback
        traceback.print_exc()


async def async_vad(websocket, audio_in):
    """
    异步 VAD (语音活动检测) 处理函数。
    检测输入音频中是否包含人声，并返回语音片段的起止时间。
    
    Args:
        websocket: WebSocket 连接对象，包含 VAD 模型的状态字典
        audio_in (bytes): 原始音频字节流
        
    Returns:
        tuple (int, int): (speech_start, speech_end)
              -1 表示未检测到开始或结束。
    """
    # 将 Bytes 转为 Tensor
    audio_tensor = decode_audio_chunk(audio_in)
    # 异步并发调用 VAD 模型
    # 注意：这里我们依旧要遵守 Nano 模型的规则（虽然是 VAD，但保持输入格式一致比较安全），传入 list
    segments_result_list = await run_model_inference(
        model_vad, input=[audio_tensor], **websocket.status_dict_vad
    )
    if not segments_result_list or len(segments_result_list) == 0:
        return -1, -1
    segments_result = segments_result_list[0]["value"]
    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


async def async_asr(websocket, audio_in):
    """
    异步离线 ASR (2pass-offline) 处理函数。
    对完整的语音片段进行高精度识别，通常在 VAD 检测到语音结束时调用。
    包含：ASR 识别 -> 标点恢复 (Punctuation Restoration) -> 发送最终结果 (is_final=True)。
    
    Args:
        websocket: WebSocket 连接对象
        audio_in (bytes): 完整的语音片段字节流
    """
    # 离线识别 (最终修正)
    if len(audio_in) > 0:
        audio_tensor = decode_audio_chunk(audio_in)
        # 异步并发调用 ASR 模型
        rec_result_list = await run_model_inference(
            model_asr, input=[audio_tensor], **websocket.status_dict_asr
        )
        if not rec_result_list or len(rec_result_list) == 0:
           # 如果为空，直接返回空文本
           rec_result = {"text": ""}
        else:
           rec_result = rec_result_list[0]
        
        # 标点恢复
        if model_punc is not None and len(rec_result["text"]) > 0:
            # 异步并发调用标点模型
            punc_result_list = await run_model_inference(
                model_punc, input=rec_result["text"], **websocket.status_dict_punc
            )
            if punc_result_list and len(punc_result_list) > 0:
                rec_result = punc_result_list[0]
        
        # 始终发送结果，即使为空，否则客户端会一直等待直到超时
        mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
        message = json.dumps(
            {
                "mode": mode,
                "text": rec_result["text"],
                "wav_name": websocket.wav_name,
                "is_final": True,
            }
        )
        try:
            await websocket.send(message)
        except Exception as e:
            # 客户端断开，安全忽略
            print(f"Client disconnected during async_asr send: {e}", flush=True)
    else:
        # Empty audio result
        mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
        message = json.dumps(
            {
                "mode": mode,
                "text": "",
                "wav_name": websocket.wav_name,
                "is_final": True,
            }
        )
        try:
            await websocket.send(message)
        except Exception as e:
            print(f"Client disconnected during async_asr empty send: {e}", flush=True)


async def async_asr_online(websocket, audio_in):
    """
    异步在线流式 ASR (Online Streaming) 处理函数。
    对实时到达的音频流进行增量识别，返回中间结果 (is_final=False)。
    
    Args:
        websocket: WebSocket 连接对象
        audio_in (bytes): 实时音频流片段
    """
    # 在线流式识别
    if len(audio_in) > 0:
        audio_tensor = decode_audio_chunk(audio_in)
        # 异步并发调用流式 ASR 模型
        rec_result_list = await run_model_inference(
             model_asr_streaming, input=[audio_tensor], **websocket.status_dict_asr_online
        )
        if not rec_result_list or len(rec_result_list) == 0:
             return
        rec_result = rec_result_list[0]
        
        if websocket.mode == "2pass" and websocket.status_dict_asr_online.get("is_final", False):
            return

        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": websocket.wav_name,
                    "is_final": not websocket.is_speaking,
                }
            )
            try:
                await websocket.send(message)
            except Exception as e:
                print(f"Client disconnected during async_asr_online send: {e}", flush=True)


async def main():
    if len(args.certfile) > 0:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_cert = args.certfile
        ssl_key = args.keyfile
        ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)
        start_server = websockets.serve(
            ws_serve, args.host, args.port, subprotocols=None, ping_interval=None, ssl=ssl_context
        )
    else:
        start_server = websockets.serve(
            ws_serve, args.host, args.port, subprotocols=None, ping_interval=None
        )
    print(f"服务已启动，监听地址: ws://{args.host}:{args.port}", flush=True)
    await start_server
    await asyncio.get_event_loop().create_future() # 永久运行


if __name__ == "__main__":
    asyncio.run(main())