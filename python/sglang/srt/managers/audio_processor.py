import concurrent.futures
import logging
import multiprocessing as mp
import os
from abc import ABC, abstractmethod


# TODO: Separate Concerns between Image and Audio Models
from sglang.srt.manager.image_processor import init_global_processor
from sglang.srt.server_args import ServerArgs

# TODO: Use this once instantiation is created
from sglang.srt.utils import load_audio

logger = logging.getLogger(__name__)


class BaseAudioProcessor(ABC):
    def __init__(self, hf_config, server_args: ServerArgs, _processor):
        self.hf_config = hf_config
        self._processor = _processor

        self.executor = concurrent.futures.ProcessPoolExecutor(
            initializer=init_global_processor,
            mp_context=mp.get_context("fork"),
            initargs=(server_args,),
            max_workers=int(os.environ.get("SGLANG_CPU_COUNT", os.cpu_count())),
        )

    @abstractmethod
    async def process_audio_async(self, audio_data, input_text, **kwargs):
        pass


class DummyAudioProcessor(BaseAudioProcessor):
    def __init__(self):
        pass

    async def process_audio_async(self, *args, **kwargs):
        return None


def get_dummy_audio_processor():
    return DummyAudioProcessor()
