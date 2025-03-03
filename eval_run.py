import torch
import torch.distributed as dist

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer
from vlmeval.vlm import LLaVA as llava_model
from llava.vlm_evalkit.llava_internlm import LLaVA_InternLM2, LLaVA_Next_InternLM2
from llava.vlm_evalkit.llava_qwen import LLaVA_Qwen2, LLaVA_Next_Qwen2
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True)
    # parser.add_argument('--model', type=str, nargs='+', required=True)
    # ==============================================
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--model', type=str, default='LLaVA_InternLM2')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--split', type=bool, default=False)
    # ==============================================
    # Args that only apply to Video Dataset
    parser.add_argument('--nframe', type=int, default=8)
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--use-subtitle', action='store_true')
    parser.add_argument('--fps', type=float, default=-1)
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Rerun: will remove all evaluation temp files
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')
    rank, world_size = get_rank_and_world_size()

    args = parse_args()
    assert len(args.data), '--data should be a list of data files'

    # ==============================================
    stop_words = ['<|im_end|>']  # internlm2
    if 'AR' in args.path or 'AR' in args.model:
        if 'internlm' in args.model.lower() or 'vicuna' in args.model.lower():
            supported_VLM[args.model] = partial(
                LLaVA_Next_InternLM2, model_path=args.path, model_base=args.model_base, split_system=args.split
            )
        elif 'qwen' in args.model.lower():
            supported_VLM[args.model] = partial(
                LLaVA_Next_Qwen2, model_path=args.path, model_base=args.model_base, split_system=args.split
            )
    else:
        if 'internlm' in args.model.lower():
            supported_VLM[args.model] = partial(
                LLaVA_InternLM2, model_path=args.path, model_base=args.model_base, split_system=args.split
            )
        elif 'qwen' in args.model.lower():
            supported_VLM[args.model] = partial(LLaVA_Qwen2, model_path=args.path, model_base=args.model_base, split_system=args.split)
    args.model = [args.model]
    # ==============================================

    if rank == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will start the evaluation from scratch')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if args.retry is not None:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600))

    for _, model_name in enumerate(args.model):
        model = None

        # pred_root = osp.join(args.work_dir, model_name)
        # os.makedirs(pred_root, exist_ok=True)

        # eval_id = timencommit()
        # pred_root = osp.join(args.work_dir, model_name, eval_id)
        pred_root_meta = osp.join(args.work_dir, model_name)
        pred_root = pred_root_meta
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        for _, dataset_name in enumerate(args.data):
            try:
                dataset_kwargs = {}
                if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                    dataset_kwargs['model'] = model_name
                if dataset_name == 'MMBench-Video':
                    dataset_kwargs['pack'] = args.pack
                if dataset_name == 'Video-MME':
                    dataset_kwargs['use_subtitle'] = args.use_subtitle

                # If distributed, first build the dataset on the main process for doing preparation works
                if world_size > 1:
                    if rank == 0:
                        dataset = build_dataset(dataset_name, **dataset_kwargs)
                        assert dataset is not None
                    dist.barrier()

                dataset = build_dataset(dataset_name, **dataset_kwargs)
                if dataset is None:
                    logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                    continue

                result_file_base = f'{model_name}_{dataset_name}.xlsx'

                # Handling Video Datasets. For Video Dataset, set the fps for priority
                if args.fps > 0:
                    if dataset_name == 'MVBench':
                        raise ValueError(
                            'MVBench does not support fps setting, please transfer to MVBench_MP4!'
                        )
                    args.nframe = 0
                if dataset_name in ['MMBench-Video']:
                    packstr = 'pack' if args.pack else 'nopack'
                    if args.nframe > 0:
                        result_file_base = f'{model_name}_{dataset_name}_{args.nframe}frame_{packstr}.xlsx'
                    else:
                        result_file_base = f'{model_name}_{dataset_name}_{args.fps}fps_{packstr}.xlsx'
                elif dataset.MODALITY == 'VIDEO':
                    if args.pack:
                        logger.info(f'{dataset_name} not support Pack Mode, directly change to unpack')
                        args.pack = False
                    packstr = 'pack' if args.pack else 'nopack'
                    if args.nframe > 0:
                        result_file_base = f'{model_name}_{dataset_name}_{args.nframe}frame_{packstr}.xlsx'
                    else:
                        result_file_base = f'{model_name}_{dataset_name}_{args.fps}fps_{packstr}.xlsx'
                    if dataset_name in ['Video-MME', 'LongVideoBench']:
                        subtitlestr = 'subs' if args.use_subtitle else 'nosubs'
                        result_file_base = result_file_base.replace('.xlsx', f'_{subtitlestr}.xlsx')

                # Handling Multi-Turn Dataset
                if dataset.TYPE == 'MT':
                    result_file_base = result_file_base.replace('.xlsx', '.tsv')

                result_file = osp.join(pred_root, result_file_base)

                if rank == 0 and len(prev_pred_roots):
                    prev_result_file = None
                    for root in prev_pred_roots[::-1]:
                        if osp.exists(osp.join(root, result_file_base)):
                            prev_result_file = osp.join(root, result_file_base)
                            break
                    if not args.reuse:
                        prev_result_file = None
                    if prev_result_file is not None:
                        logger.warning(f'--reuse is set, will reuse the prediction file {prev_result_file}.')
                        if prev_result_file != result_file:
                            shutil.copy(prev_result_file, result_file)

                if world_size > 1:
                    dist.barrier()

                if model is None:
                    model = model_name  # which is only a name

                # Perform the Inference
                if dataset.MODALITY == 'VIDEO':
                    model = infer_data_job_video(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        nframe=args.nframe,
                        pack=args.pack,
                        verbose=args.verbose,
                        subtitle=args.use_subtitle,
                        api_nproc=args.nproc,
                        fps=args.fps,
                    )
                elif dataset.TYPE == 'MT':
                    model = infer_data_job_mt(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.nproc,
                        ignore_failed=args.ignore,
                    )
                else:
                    model = infer_data_job(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.nproc,
                        ignore_failed=args.ignore,
                    )

                # Set the judge kwargs first before evaluation or dumping

                judge_kwargs = {
                    'nproc': args.nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                }

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge
                else:
                    if dataset.TYPE in ['MCQ', 'Y/N'] or listinstr(['MathVerse'], dataset_name):
                        judge_kwargs['model'] = 'chatgpt-0125'
                    elif listinstr(
                        ['MMVet', 'MathVista', 'LLaVABench', 'MMBench-Video', 'MathVision'], dataset_name
                    ):
                        judge_kwargs['model'] = 'gpt-4-turbo'
                    elif listinstr(
                        [
                            'MMLongBench',
                            'MMDU',
                            'DUDE',
                            'SLIDEVQA',
                            'MIA-Bench',
                            'WildVision',
                            'MMAlignBench',
                        ],
                        dataset_name,
                    ):
                        judge_kwargs['model'] = 'gpt-4o'
                    # elif listinstr(['MMVet'], dataset_name):
                    #     judge_kwargs['model'] = 'gpt-4o-0806'
                logger.info(judge_kwargs)

                if rank == 0:
                    if dataset_name in ['MMMU_TEST']:
                        result_json = MMMU_result_transfer(result_file)
                        logger.info(
                            f'Transfer MMMU_TEST result to json for official evaluation, '
                            f'json file saved in {result_json}'
                        )  # noqa: E501
                        continue
                    elif 'MMT-Bench_ALL' in dataset_name:
                        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                        logger.info(
                            f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                            f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                            f'submission file saved in {submission_file}'
                        )  # noqa: E501
                        continue
                    elif 'MLLMGuard_DS' in dataset_name:
                        logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')  # noqa: E501
                        continue
                    elif 'AesBench_TEST' == dataset_name:
                        logger.info(
                            f'The results are saved in {result_file}. '
                            f'Please send it to the AesBench Team via huangyipo@hotmail.com.'
                        )  # noqa: E501
                        continue
                    elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                        logger.info(
                            f'{dataset_name} is a test split without ground-truth. '
                            'Thus only the inference part is supported for those datasets. '
                        )  # noqa: E501

                if dataset_name in [
                    'MMBench_TEST_CN',
                    'MMBench_TEST_EN',
                    'MMBench',
                    'MMBench_CN',
                    'MMBench_TEST_CN_V11',
                    'MMBench_TEST_EN_V11',
                    'MMBench_V11',
                    'MMBench_CN_V11',
                ]:
                    if not MMBenchOfficialServer(dataset_name):
                        logger.error(
                            f'Can not evaluate {dataset_name} on non-official servers, '
                            'will skip the evaluation. '
                        )
                        continue

                eval_proxy = os.environ.get('EVAL_PROXY', None)
                old_proxy = os.environ.get('HTTP_PROXY', '')

                if world_size > 1:
                    dist.barrier()

                if rank == 0 and args.mode == 'all':
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        logger.info(
                            f'The evaluation of model {model_name} x dataset {dataset_name} has finished! '
                        )
                        logger.info('Evaluation Results:')
                    if isinstance(eval_results, dict):
                        logger.info('\n' + json.dumps(eval_results, indent=4))
                    elif isinstance(eval_results, pd.DataFrame):
                        if len(eval_results) < len(eval_results.columns):
                            eval_results = eval_results.T
                        logger.info('\n' + tabulate(eval_results))

                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    files = os.listdir(pred_root)
                    files = [x for x in files if f'{model_name}_{dataset_name}' in x]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        # if osp.exists(link_addr) or osp.islink(link_addr):
                        #     os.remove(link_addr)
                        # os.symlink(file_addr, link_addr)

            except Exception as e:
                logger.exception(
                    f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                    'skipping this combination.'
                )
                continue

            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
