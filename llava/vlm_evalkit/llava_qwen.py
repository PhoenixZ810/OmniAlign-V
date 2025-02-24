import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from vlmeval.vlm.base import BaseModel
from vlmeval.smp import *
from vlmeval.dataset import DATASET_TYPE
import copy


class LLaVA_Qwen2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='liuhaotian/llava_v1.5_7b', model_base=None, split_system=False, **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install llava before using LLaVA')
            sys.exit(-1)

        warnings.warn('Please install the latest version of llava from github before you evaluate the LLaVA model. ')
        assert osp.exists(model_path) or splitlen(model_path) == 2
        if not split_system:
            self.system_prompt = "<|im_start|>system\nYou are a helpful assistant."
        else:
            self.system_prompt = "<|im_start|>system\nYou are a helpful assistant, you can understand the visual content that the user provides, and assist the user with a variety of tasks using natural language. "
        self.stop_str = '<|im_end|>'

        if model_path == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_path == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_path)

        # try:
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            device='cpu',
            device_map='cpu',
            trust_remote_code=True,
        )
        # except:
        #     if 'ShareGPT4V' in model_path:
        #         import llava
        #         warnings.warn(
        #             'Please manually remove the encoder type check in '
        #             f'{llava.__path__[0]}/model/multimodal_encoder/builder.py '
        #             'Line 8 to use the ShareGPT4V model. ')
        #     else:
        #         warnings.warn('Unknown error when loading LLaVA model.')
        #     exit(-1)

        self.model = self.model.cuda()

        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True
        )  # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。'
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                # text += '<image>\n'
                images.append(item['value'])
        return text, images

    def chat_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += 'USER: ' if utter['role'] == 'user' else 'ASSISTANT: '
            content, images_sub = self.concat_tilist(utter['content'])
            prompt += content
            images.extend(images_sub)
            prompt += ' ' if utter['role'] == 'user' else self.stop_str
        assert message[-1]['role'] == 'user', message
        prompt += 'ASSISTANT: '

        images = [Image.open(s).convert('RGB') for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs
            )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert('RGB') for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        if images:
            image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)
        else:
            image_tensor = None

        prompt = self.system_prompt + '<|im_end|>\n<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n'

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        )
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output


class LLaVA_Next_Qwen2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    # This function is used to split InternVL2-Llama3-76B
    def split_model(self, model_path):
        import math

        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size
        if "72b" not in model_path.lower():
            return None
        # embed_tokens, vision_tower, mm_projector, lm_head are treated as 2 layers
        num_layers = 80 + 8
        num_layers_per_gpu = math.ceil(num_layers / num_gpus)
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        num_layers_per_gpu[0] -= 6
        num_layers_per_gpu[-1] -= 2
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f"model.layers.{layer_cnt}"] = rank + world_size * i
                layer_cnt += 1
        last_gpu = rank + world_size * (num_gpus - 1)
        device_map["model.image_newline"] = rank
        device_map["model.embed_tokens"] = rank
        device_map["model.norm"] = rank
        device_map["model.vision_tower"] = rank
        device_map["model.vision_resampler"] = rank
        device_map["model.mm_projector"] = rank
        device_map["lm_head"] = last_gpu
        return device_map

    def __init__(self, model_path="lmms-lab/llava-onevision-qwen2-7b-si", split_system=False, **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
                KeywordsStoppingCriteria,
            )  # noqa: E501
        except Exception as err:
            logging.critical("Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`")
            raise err

        video_kwargs_default = dict(overwrite=True, mm_spatial_pool_mode="average", force_sample=True)
        video_kwargs_default.update(kwargs)
        self.video_kwargs = video_kwargs_default

        overwrite_config = None
        if "video" in model_path.lower():
            if self.video_kwargs["overwrite"]:
                overwrite_config = {}
                overwrite_config["mm_spatial_pool_mode"] = self.video_kwargs["mm_spatial_pool_mode"]

        rank, world_size = get_rank_and_world_size()
        model_name = get_model_name_from_path(model_path)
        device_map = self.split_model(model_path)

        if device_map is None:
            if auto_split_flag():
                assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT set for non-72B LLaVA-OneVision'
                logging.warning('Currently, we only support to split the non-72B model across all GPUs.')
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path,
                    None,
                    model_name,
                    device_map="auto",
                    overwrite_config=overwrite_config,
                )
            else:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path,
                    None,
                    model_name,
                    device_map="cpu",
                    # overwrite_config=overwrite_config,
                )
                model.cuda()
        else:
            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path,
                None,
                model_name,
                device_map=device_map,
                overwrite_config=overwrite_config,
            )
        model.eval()
        model.tie_weights()
        if rank == 0:
            print(model)

        if 'intern' in model_path.lower():
            conv_mode = "internlm_2"
        elif 'vicuna' in model_path.lower():
            conv_mode = "v1"
        elif "llava" in model_path.lower():
            conv_mode = "qwen_1_5"
        if 'llava-video' in model_path.lower():
            self.nframe = 64
        else:
            self.nframe = 16
            if "72b" in model_path.lower():
                self.nframe = 32

        if "video" in model_path.lower():
            self.force_sample = self.video_kwargs["force_sample"]
        else:
            self.force_sample = False

        if not split_system:
            self.system_prompt = "<|im_start|>system\nYou are a helpful assistant."
        else:
            self.system_prompt = "<|im_start|>system\nYou are a helpful assistant, you can understand the visual content that the user provides, and assist the user with a variety of tasks using natural language. "
        self.stop_str = '<|im_end|>'

        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images  # Store process_images as a class attribute
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

        image_aspect_ratio = getattr(self.model.config, "image_aspect_ratio", None)
        image_grid_pinpoints = getattr(self.model.config, "image_grid_pinpoints", None)
        print('----------------------------------------------------------------')
        print('The split strategy for the model is:')
        print(f"image_aspect_ratio: {image_aspect_ratio}")
        print(f"image_grid_pinpoints: {image_grid_pinpoints}")
        print('----------------------------------------------------------------')

    def generate_inner(self, message, dataset=None):
        content, images = "", []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                # content += self.DEFAULT_IMAGE_TOKEN + "\n"
                content = self.DEFAULT_IMAGE_TOKEN + "\n" + content

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]

        # conv = copy.deepcopy(self.conv_templates[self.conv_template])
        # conv.append_message(conv.roles[0], content)
        # conv.append_message(conv.roles[1], None)
        # prompt_question = conv.get_prompt()
        prompt = self.system_prompt + '<|im_end|>\n<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n'

        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [self.stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_chat(self, message, temperature=0, max_new_tokens=512):
        content, images = "", []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg["type"] == "text":
                if self.DEFAULT_IMAGE_TOKEN in msg["value"] and self.DEFAULT_IMAGE_TOKEN in content:
                    msg["value"] = msg["value"].replace(self.DEFAULT_IMAGE_TOKEN + '\n', '')
                    msg["value"] = msg["value"].replace(self.DEFAULT_IMAGE_TOKEN, '')
                content += msg["value"]
            else:
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content += self.DEFAULT_IMAGE_TOKEN + "\n"
        # print(content)
        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=True,
            temperature=temperature,
            top_k=10,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_with_imfeature(self, message, image_file_path, temperature=0, max_new_tokens=512):
        content, images = "", []
        image_sizes = []  # Store image sizes

        # print(content)
        # Process images using the class attribute self.process_images
        # image_tensor = self.process_images(images, self.image_processor, self.model.config)
        # image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
        original_pixel_values = torch.load(image_file_path, map_location="cpu")
        image_tensor = original_pixel_values.cuda()
        image_sizes = torch.tensor([[899, 1024]]).cuda()

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        prompt_question = message

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Pass image sizes along with other parameters
        # cont = self.model.generate(
        #     input_ids,
        #     images=image_tensor,
        #     image_sizes=image_sizes,  # Pass the image sizes here
        #     do_sample=False,
        #     temperature=temperature,
        #     top_k=10,
        #     top_p=0.9,
        #     max_new_tokens=max_new_tokens,
        #     stopping_criteria=[stopping_criteria],
        # )
        cont = self.model(input_ids, images=image_tensor, image_sizes=image_sizes)
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs
