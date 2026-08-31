# Train RNN on motorTraj_* control tasks (classic RNN, no STP).
# By Shanglin Zhou



import numpy as np
import torch

import time
import os
import random
import sys
from utils import save_checkpoint, load_checkpoint


Mode = 'eval' # eval, train

# Current training task. The environment override avoids editing this file:
# $env:RNN_TASK='motorTraj_circle_time'; python code\PARAM.py
SUPPORTED_TASKS = (
    'motorTraj_circle',
    'motorTraj_circle_time',
    'motorTraj_circle_space',
)
SINGLE_AXIS_TASKS = frozenset((
    'motorTraj_circle_time',
    'motorTraj_circle_space',
))
SELECTED_TASK = os.environ.get('RNN_TASK', 'motorTraj_circle_space')
if SELECTED_TASK not in SUPPORTED_TASKS:
    raise ValueError(
        'Unsupported task {!r}; expected one of {}'.format(SELECTED_TASK, SUPPORTED_TASKS)
    )
if SELECTED_TASK in SINGLE_AXIS_TASKS:
    # Checkpoints may have been serialized with NumPy 2.x module paths while
    # this evaluation environment exposes the NumPy 1.x aliases.
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)

# Systematic three-interface sweep.
# W_in: cue-to-control overlap; W_rec: recurrent capacity/rank; W_out: fixed/free output plane.
INPUT_OVERLAP_SWEEP = [-0.8,  -0.4,  0.0, 0.4,  0.8]   #[-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8] 
RANK_SWEEP = [None]  # None is full rank.
READOUT_MODE_SWEEP = ['free','fixed'] #READOUT_MODE_SWEEP = ['free', 'fixed']
if SELECTED_TASK == 'motorTraj_circle':
    SWEEP_CONFIGS = [
        (input_overlap_target, rank_value, readout_mode)
        for input_overlap_target in INPUT_OVERLAP_SWEEP
        for rank_value in RANK_SWEEP
        for readout_mode in READOUT_MODE_SWEEP
    ]
else:
    # Single-axis controls use only the prespecified canonical configuration:
    # speed-coded, full-rank, input overlap -0.8, and free readout.
    SWEEP_CONFIGS = [(-0.8, None, 'free')]
numPara = len(SWEEP_CONFIGS)

BASE_SEED = 20260626
JOINT_LOSS_CHECKPOINT_THRESHOLDS = [0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125]
SINGLE_AXIS_LOSS_CHECKPOINT_THRESHOLDS = [
    round(0.30 - 0.01 * index, 2) for index in range(26)
]
LOSS_CHECKPOINT_THRESHOLDS = (
    SINGLE_AXIS_LOSS_CHECKPOINT_THRESHOLDS
    if SELECTED_TASK in SINGLE_AXIS_TASKS
    else JOINT_LOSS_CHECKPOINT_THRESHOLDS
)
TRAIN_EVAL_FREQ = 5 if SELECTED_TASK in SINGLE_AXIS_TASKS else 100
TRAIN_LOSS_STOP = 0.05 if SELECTED_TASK in SINGLE_AXIS_TASKS else 0.125
EVAL_ON_BATCH_MULTIPLE = bool(SELECTED_TASK in SINGLE_AXIS_TASKS)
INCLUSIVE_LOSS_STOP = bool(SELECTED_TASK in SINGLE_AXIS_TASKS)


# Defaults for this sweep: speed-coded time input and narrow input range.
# Set USE_SPEED_CODED_TIME_INPUT = False for duration-coded inputs. The same
# input levels are used, but their target-duration mapping is reversed.
USE_SPEED_CODED_TIME_INPUT = (
    False if SELECTED_TASK == 'motorTraj_circle' else True
)
TIME_CODE_TAG = 'speedcoded' if USE_SPEED_CODED_TIME_INPUT else 'durationcoded'

USE_NARROW_INPUT_LEVELS = True
NARROW_INPUT_LEVELS = [0.4, 0.5, 0.6, 0.7, 0.8]
NARROW_INTERP_LEVELS = [0.45, 0.55, 0.65, 0.75]
NARROW_EXTRA_LOW_LEVELS = [0.35, 0.3]
NARROW_EXTRA_HIGH_LEVELS = [0.85, 0.9]
INPUT_LEVEL_TAG = 'narrow0408_fulltarget'

# Default single-axis evaluation checkpoints use the task-specific thresholds
# selected for controlled-output matching to the corresponding joint-task
# central slices: 0.12 for duration-only and 0.08 for radius-only. The joint
# task retains its historical 0.15 default. RNN_EVAL_LOSS_CKPT remains an
# explicit override for checkpoint-sensitivity analyses.
if SELECTED_TASK == 'motorTraj_circle_time':
    DEFAULT_EVAL_LOSS_CHECKPOINT_THRESHOLD = 0.12
elif SELECTED_TASK == 'motorTraj_circle_space':
    DEFAULT_EVAL_LOSS_CHECKPOINT_THRESHOLD = 0.08
else:
    DEFAULT_EVAL_LOSS_CHECKPOINT_THRESHOLD = 0.15
EVAL_LOSS_CHECKPOINT_THRESHOLD = float(os.environ.get(
    'RNN_EVAL_LOSS_CKPT',
    DEFAULT_EVAL_LOSS_CHECKPOINT_THRESHOLD,
))
# Toggle this directly for eval payload size.
# False: do not save recurrent firing rates; eval filename gets "_NoRates".
# True: save recurrent firing rates; eval filename keeps the current rates format.
SAVE_EVAL_RATES = True

# Device selection: 'cpu', 'cuda', 'cuda:0', or 'auto'. Can be overridden from PowerShell:
# $env:RNN_DEVICE='cuda'; python code\PARAM.py
RUN_DEVICE = os.environ.get('RNN_DEVICE', 'cpu')

# Low-rank eval dynamics files are large. Keep checkpoints and full-rank results
# under the project results directory, but redirect low-rank eval payloads here.
LOWRANK_EVAL_DYNAMICS_DIR = os.environ.get(
    'LOWRANK_EVAL_DYNAMICS_DIR',
    r'E:\project\Spatial-temporal scale control\code\results\motorTraj_circle',
)


def set_global_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def rank_tag(rank_value):
    return 'Rank_full' if rank_value is None else 'Rank_{}'.format(int(rank_value))


def overlap_tag(value):
    value = float(value)
    if abs(value) < 1e-12:
        return 'Overlap_0'
    sign = 'p' if value > 0 else 'm'
    digits = '{:.2f}'.format(abs(value)).replace('.', '')
    return 'Overlap_{}'.format(sign + digits)


def readout_tag(mode):
    return 'Readout_{}'.format(str(mode).lower())


def time_code_tag(value):
    return 'Time_{}'.format(str(value).lower())


def eval_rates_file_suffix(hp):
    return '' if bool(hp.get('save_eval_rates', False)) else '_NoRates'


def loss_tag(value):
    return '{:.3f}'.format(float(value)).replace('.', 'p')


def result_output_dir(default_out_dir, hp):
    if Mode == 'eval' and hp.get('rank') is not None:
        return LOWRANK_EVAL_DYNAMICS_DIR
    return default_out_dir


for paraInd in range(numPara):
    input_overlap_target, para, readout_mode = SWEEP_CONFIGS[paraInd]
    for repInd in range(0, 20):
        seed = BASE_SEED + int(repInd)
        set_global_seed(seed)

        from model import RNN_Dale

        from tasks import (
            generate_input_motorTraj_circle,
            generate_target_motorTraj_circle,
            list_eval_conditions,
        )

        if para is None:
            model_rnn_type = 'full_rank'
            model_rank = None
        else:
            model_rnn_type = 'low_rank'
            model_rank = int(para)

        hp = {
            'task': SELECTED_TASK,
            'device': RUN_DEVICE,

            'train_w_flag': True,
            'train_wout_flag': str(readout_mode).lower() == 'free',
            'train_win_flag': True,

            'N': 200,
            'num_In': 3,
            'num_Out': 2,
            'gain': 0.5,
            'input_gain': 1.0,
            'apply_dale': False,
            'P_rec': 1,
            'P_inh': 0.2,
            'w_dist': 'gaus',
            'rnn_type': model_rnn_type,
            'rank': model_rank,

            'activation': 'tanh',
            'dynamics_mode': 'current_mode',

            'dt': 20,
            'tau_rnn': 100.0,

            # Training / eval schedule defaults.
            'learning_rate': 0.001,
            'n_trials': 30000,
            'eval_freq': 100,
            'batch_size': 16,
            'train_eval_batches': 20,
            'loss_thr': 0.1,# default 0.125 for joint task, 0.1 for single-axis time tasks,0.07 for single-axis space tasks
            'perf_thr': 0.0,
            'eval_tr': 20,
            'save_every': 1000,
            'save_best': True,
            'resume_path': '',
            'eval_ckpt_path': '',
            'eval_loss_checkpoint_threshold': float(EVAL_LOSS_CHECKPOINT_THRESHOLD),
            'strict_load': True,
            'export_mat': False,
            'save_train_eval_dynamics': False,
            'save_eval_rates': SAVE_EVAL_RATES,
            'save_eval_outputs_net': False,
            'input_set': list(NARROW_INPUT_LEVELS) if USE_NARROW_INPUT_LEVELS else [0.2, 0.4, 0.6, 0.8, 1.0],
            'input_level_tag': INPUT_LEVEL_TAG,
            'seed': int(seed),
            'sweep_index': int(paraInd),
            'sweep_input_overlap': float(input_overlap_target),
            'sweep_rank': 'full' if para is None else int(para),
            'sweep_readout_mode': str(readout_mode).lower(),
            'enforce_input_overlap': True,
            'input_overlap_target': float(input_overlap_target),
            'output_readout_mode': str(readout_mode).lower(),
            'fixed_readout_scale': 1.0,
            'loss_checkpoint_thresholds': list(LOSS_CHECKPOINT_THRESHOLDS),
            'eval_on_batch_multiple': EVAL_ON_BATCH_MULTIPLE,
            'inclusive_loss_stop': INCLUSIVE_LOSS_STOP,
            'circle_target_map_levels': list(NARROW_INPUT_LEVELS) if USE_NARROW_INPUT_LEVELS else [0.2, 0.4, 0.6, 0.8, 1.0],
            'eval_train_levels': list(NARROW_INPUT_LEVELS) if USE_NARROW_INPUT_LEVELS else [0.2, 0.4, 0.6, 0.8, 1.0],
            'eval_interp_levels': list(NARROW_INTERP_LEVELS) if USE_NARROW_INPUT_LEVELS else [0.3, 0.5, 0.7, 0.9],
            'eval_extra_low_levels': list(NARROW_EXTRA_LOW_LEVELS) if USE_NARROW_INPUT_LEVELS else [0.1],
            'eval_extra_high_levels': list(NARROW_EXTRA_HIGH_LEVELS) if USE_NARROW_INPUT_LEVELS else [1.1, 1.2],
            'eval_allow_extrapolate': True,
            'eval_grid_name': 'narrow0408_fulltarget_train_interp_extra_13x13' if USE_NARROW_INPUT_LEVELS else 'train_interp_extra_12x12',
            'use_speed_coded_time_input': bool(USE_SPEED_CODED_TIME_INPUT),
            'time_code_tag': TIME_CODE_TAG,
            'mechanism_size_input_channel': 1,
            'mechanism_speed_input_channel': 2,
        }

        if SELECTED_TASK == 'motorTraj_circle_time':
            hp['circle_fixed_size_level'] = float(np.median(hp['input_set']))
        elif SELECTED_TASK == 'motorTraj_circle_space':
            hp['circle_fixed_time_level'] = float(np.median(hp['input_set']))

        hp['N'] = 200
        hp['stim_on'] = np.int32(600 / hp['dt'])
        hp['stim_dur'] = np.int32(200 / hp['dt'])
        hp['sigma'] = 0.01
        hp['b_rec0'] = np.zeros((hp['N'], 1))
        hp['b_out0'] = np.zeros((hp['num_Out'], 1))

        # motorTraj_circle hyperparameters.
        hp['learning_rate'] = 0.001
        hp['n_trials'] = 30000
        hp['eval_freq'] = TRAIN_EVAL_FREQ
        hp['batch_size'] = 16
        hp['train_eval_batches'] = 20
        hp['eval_tr'] = 20
        hp['loss_thr'] = TRAIN_LOSS_STOP
        hp['perf_thr'] = 1
        max_move_ms = 3800.0 if USE_NARROW_INPUT_LEVELS else 3100.0
        if SELECTED_TASK == 'motorTraj_circle_time':
            hp['task_note'] = 'clockwise circle; duration control; radius fixed at center cue'
        elif SELECTED_TASK == 'motorTraj_circle_space':
            hp['task_note'] = 'clockwise circle; radius control; duration fixed at center cue'
        else:
            hp['task_note'] = 'clockwise circle; joint radius-duration control'

        if SELECTED_TASK == 'motorTraj_circle_time':
            hp['eval_grid_name'] = hp['eval_grid_name'].replace('13x13', '1x13').replace('12x12', '1x12')
        elif SELECTED_TASK == 'motorTraj_circle_space':
            hp['eval_grid_name'] = hp['eval_grid_name'].replace('13x13', '13x1').replace('12x12', '12x1')

        hp['T'] = np.int32(600 / hp['dt']) + hp['stim_dur'] + np.int32(np.ceil(max_move_ms / hp['dt']))

        #print(hp)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, 'results', hp['task'])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        run_id = '{}_{}_{}_{}_rep{:02d}'.format(
            rank_tag(para),
            overlap_tag(input_overlap_target),
            readout_tag(readout_mode),
            time_code_tag(TIME_CODE_TAG),
            int(repInd),
        )
        legacy_speedcoded_run_ids = []
        if USE_SPEED_CODED_TIME_INPUT:
            legacy_speedcoded_run_ids = [
                '{}_{}_{}_rep{:02d}'.format(
                    rank_tag(para),
                    overlap_tag(input_overlap_target),
                    readout_tag(readout_mode),
                    int(repInd),
                ),
                '{}_{}_{}_{}_rep{:02d}'.format(
                    rank_tag(para),
                    overlap_tag(input_overlap_target),
                    readout_tag(readout_mode),
                    time_code_tag(''),
                    int(repInd),
                ),
                '{}_{}_{}{}_rep{:02d}'.format(
                    rank_tag(para),
                    overlap_tag(input_overlap_target),
                    readout_tag(readout_mode),
                    time_code_tag(''),
                    int(repInd),
                ),
            ]
        ckpt_dir = os.path.join(out_dir, 'checkpoints', run_id)
        os.makedirs(ckpt_dir, exist_ok=True)
        last_ckpt_path = os.path.join(ckpt_dir, 'last.pt')
        best_ckpt_path = os.path.join(ckpt_dir, 'best.pt')
        eval_loss_ckpt_path = os.path.join(
            ckpt_dir,
            'loss_{}.pt'.format(loss_tag(hp.get('eval_loss_checkpoint_threshold', 0.15))),
        )
        legacy_eval_loss_ckpt_paths = [
            os.path.join(
                out_dir,
                'checkpoints',
                legacy_run_id,
                'loss_{}.pt'.format(loss_tag(hp.get('eval_loss_checkpoint_threshold', 0.15))),
            )
            for legacy_run_id in legacy_speedcoded_run_ids
        ]
        loss_checkpoint_values = [float(v) for v in hp.get('loss_checkpoint_thresholds', [])]
        loss_ckpt_paths = {
            float(v): os.path.join(ckpt_dir, 'loss_{}.pt'.format(loss_tag(v)))
            for v in loss_checkpoint_values
        }

        if Mode == 'train':
            stale_stubs = [
                '{}_train'.format(run_id),
                '{}_eval_{}'.format(run_id, hp.get('eval_grid_name', 'grid')),
            ]
            stale_paths = [last_ckpt_path, best_ckpt_path] + list(loss_ckpt_paths.values())
            if SELECTED_TASK in SINGLE_AXIS_TASKS:
                stale_paths.extend(
                    os.path.join(ckpt_dir, name)
                    for name in os.listdir(ckpt_dir)
                    if name.startswith('loss_') and name.endswith('.pt')
                )
            for stale_stub in stale_stubs:
                stale_paths.append(os.path.join(out_dir, stale_stub + '.pt'))
                stale_paths.append(os.path.join(out_dir, stale_stub + '.mat'))
            for stale_path in stale_paths:
                if os.path.exists(stale_path):
                    try:
                        os.remove(stale_path)
                    except PermissionError:
                        print('Warning: could not remove locked stale file: {}'.format(stale_path))
            print('Cleared previous outputs for {}'.format(run_id))

        rnn = RNN_Dale(hp)
        set_global_seed(seed)

        def train_step(stim, label, target, target_mask):
            optimizer.zero_grad()
            t_r, t_o, t_o_net, t_loss = rnn(stim, label, target, target_mask)
            t_loss.backward()
            clipmin = -10
            clipmax = 10
            for param in rnn.train_var:
                if param.grad is None:
                    continue
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                param.grad.data.clamp_(clipmin, clipmax)
            optimizer.step()
            rnn.enforce_input_overlap_()
            return t_r, t_o, t_o_net, t_loss

        eval_conditions = list_eval_conditions(hp)

        def generate_batch_by_task(mode='train', condition_idx=0, trial_idx=0):
            if mode == 'train':
                hp['stim_on'] = np.int32(np.random.random_sample() * 400 / hp['dt'] + 200 / hp['dt'])
            elif mode == 'eval':
                hp['stim_on'] = np.int32(np.random.random_sample() * 400 / hp['dt'] + 200 / hp['dt'])
            eval_cond = None
            bs_arg = None
            if mode == 'eval':
                eval_cond = eval_conditions[condition_idx]
                bs_arg = 1
            stim, label = generate_input_motorTraj_circle(
                hp, mode=mode, eval_conditions=eval_cond, batch_size=bs_arg)
            target, target_mask = generate_target_motorTraj_circle(
                hp, label, batch_size=bs_arg)
            if isinstance(target, np.ndarray):
                if target.ndim == 1:
                    target = target[None, None, :]
                elif target.ndim == 2:
                    target = target[:, None, :]
            if isinstance(target_mask, np.ndarray):
                if target_mask.ndim == 1:
                    target_mask = target_mask[None, None, :]
                elif target_mask.ndim == 2:
                    target_mask = target_mask[:, None, :]
            return stim, label, target, target_mask

        def _perf_for_batch(label, t_out_np, bs):
            return np.ones((1, bs))

        def run_evaluation():
            last_stim, last_target, last_target_mask = None, None, None
            collect_rates = bool(hp.get('save_eval_rates', False))
            collect_outputs_net = bool(hp.get('save_eval_outputs_net', False))
            if Mode == 'train':
                n_mon = hp['train_eval_batches']
                bs = hp['batch_size']
                eval_perfs = np.zeros((n_mon, 1))
                eval_losses = np.zeros((n_mon, 1))
                eval_os = np.zeros((n_mon, hp['num_Out'], bs, hp['T']), dtype=np.float32)
                eval_o_nets = (np.zeros((n_mon, hp['num_Out'], bs, hp['T']), dtype=np.float32)
                               if collect_outputs_net else None)
                eval_us = np.zeros((n_mon, hp['num_In'], bs, hp['T']), dtype=np.float32)
                eval_zs = np.zeros((n_mon, hp['num_Out'], bs, hp['T']), dtype=np.float32)
                eval_ms = np.zeros((n_mon, hp['num_Out'], bs, hp['T']), dtype=np.float32)
                eval_rs = (np.zeros((n_mon, hp['N'], bs, hp['T']), dtype=np.float32)
                           if collect_rates else None)
                for ii in range(n_mon):
                    stim, label, target, target_mask = generate_batch_by_task(mode='train')
                    t_r, t_out, t_out_net, t_loss = rnn(stim, label, target, target_mask)
                    t_out_np = t_out.detach().cpu().numpy()
                    t_perf = _perf_for_batch(label, t_out_np, bs)
                    eval_losses[ii, 0] = t_loss.item()
                    eval_perfs[ii, 0] = np.mean(t_perf)
                    eval_os[ii, :, :, :] = t_out_np
                    if collect_outputs_net:
                        eval_o_nets[ii, :, :, :] = t_out_net.detach().cpu().numpy()
                    eval_us[ii, :, :, :] = stim
                    eval_zs[ii, :, :, :] = target
                    eval_ms[ii, :, :, :] = np.asarray(target_mask)
                    if collect_rates:
                        eval_rs[ii, :, :, :] = t_r.detach().cpu().numpy()
                    last_stim, last_target, last_target_mask = stim, target, target_mask
            else:
                n_cond = len(eval_conditions)
                nt = hp['eval_tr']
                bs = 1
                eval_perfs = np.zeros((n_cond, nt))
                eval_losses = np.zeros((n_cond, nt))
                eval_os = np.zeros((n_cond, nt, hp['num_Out'], hp['T']), dtype=np.float32)
                eval_o_nets = (np.zeros((n_cond, nt, hp['num_Out'], hp['T']), dtype=np.float32)
                               if collect_outputs_net else None)
                eval_us = np.zeros((n_cond, nt, hp['num_In'], hp['T']), dtype=np.float32)
                eval_zs = np.zeros((n_cond, nt, hp['num_Out'], hp['T']), dtype=np.float32)
                eval_ms = np.zeros((n_cond, nt, hp['num_Out'], hp['T']), dtype=np.float32)
                eval_rs = (np.zeros((n_cond, nt, hp['N'], hp['T']), dtype=np.float32)
                           if collect_rates else None)
                for ci in range(n_cond):
                    for ti in range(nt):
                        stim, label, target, target_mask = generate_batch_by_task(
                            mode='eval', condition_idx=ci, trial_idx=ti)
                        t_r, t_out, t_out_net, t_loss = rnn(stim, label, target, target_mask)
                        t_out_np = t_out.detach().cpu().numpy()
                        t_perf = _perf_for_batch(label, t_out_np, bs)
                        eval_losses[ci, ti] = t_loss.item()
                        eval_perfs[ci, ti] = np.mean(t_perf)
                        eval_os[ci, ti, :, :] = t_out_np[:, 0, :]
                        if collect_outputs_net:
                            eval_o_nets[ci, ti, :, :] = t_out_net.detach().cpu().numpy()[:, 0, :]
                        eval_us[ci, ti, :, :] = stim[:, 0, :]
                        eval_zs[ci, ti, :, :] = target[:, 0, :]
                        eval_ms[ci, ti, :, :] = np.asarray(target_mask)[:, 0, :]
                        if collect_rates:
                            eval_rs[ci, ti, :, :] = t_r.detach().cpu().numpy()[:, 0, :]
                        last_stim, last_target, last_target_mask = stim, target, target_mask

            eval_loss_mean = float(np.nanmean(eval_losses))
            eval_perf_mean = float(np.nanmean(eval_perfs))
            return (eval_loss_mean, eval_perf_mean, eval_losses, eval_perfs, eval_os, eval_o_nets, eval_us,
                    eval_zs, eval_rs, eval_ms, last_stim, last_target, last_target_mask)

        if Mode == 'train':
            print('TRAINING STARTED ...')
            optimizer = torch.optim.Adam(rnn.train_var, lr=hp['learning_rate'])
            best_eval_loss = np.inf
            start_trial = 0

            if hp['resume_path']:
                ckpt = load_checkpoint(hp['resume_path'], rnn, optimizer=optimizer,
                                       load_optimizer=True, strict=hp['strict_load'])
                start_trial = int(ckpt.get('step', 0))
                if ckpt.get('best_metric') is not None:
                    best_eval_loss = ckpt['best_metric']
                print('Resumed from checkpoint: {}, step={}'.format(hp['resume_path'], start_trial))

            losses = np.zeros((hp['n_trials'],))
            perfEvals = np.zeros((hp['n_trials'],))
            lossEvals = np.zeros((hp['n_trials'],))
            evalCount = 0
            reached_loss_checkpoints = {}
            start_time = time.time()

            for tr in range(start_trial, hp['n_trials']):
                stim, label, target, target_mask = generate_batch_by_task(mode='train')
                t_r, t_o, t_o_net, t_loss = train_step(stim, label, target, target_mask)
                losses[tr] = t_loss.item()

                if hp.get('eval_on_batch_multiple', False):
                    should_evaluate = (tr + 1) % hp['eval_freq'] == 0
                else:
                    # Preserve the historical joint-task schedule exactly.
                    should_evaluate = (tr - 1) % hp['eval_freq'] == 0

                if should_evaluate:
                    eval_loss_mean, eval_perf_mean, eval_losses, eval_perfs, eval_os, eval_o_nets, eval_us, \
                        eval_zs, eval_rs, _, eval_stim, eval_target, eval_target_mask = run_evaluation()
                    perfEvals[evalCount] = eval_perf_mean
                    lossEvals[evalCount] = eval_loss_mean
                    evalCount = evalCount + 1
                    stim, target, target_mask = eval_stim, eval_target, eval_target_mask

                    print("Trial = %5d, Loss: %.4f, Perf: %.4f" % (tr, eval_loss_mean, eval_perf_mean))

                    if hp['save_best'] and eval_loss_mean < best_eval_loss:
                        best_eval_loss = eval_loss_mean
                        save_checkpoint(best_ckpt_path, rnn, optimizer=optimizer, step=tr + 1,
                                        best_metric=float(best_eval_loss), hp=hp)

                    for loss_checkpoint_value in loss_checkpoint_values:
                        loss_checkpoint_key = loss_tag(loss_checkpoint_value)
                        if (loss_checkpoint_key not in reached_loss_checkpoints
                                and eval_loss_mean <= float(loss_checkpoint_value)):
                            save_checkpoint(
                                loss_ckpt_paths[float(loss_checkpoint_value)],
                                rnn,
                                optimizer=optimizer,
                                step=tr + 1,
                                best_metric=float(eval_loss_mean),
                                hp=hp,
                            )
                            reached_loss_checkpoints[loss_checkpoint_key] = {
                                'threshold': float(loss_checkpoint_value),
                                'step': int(tr + 1),
                                'eval_loss': float(eval_loss_mean),
                                'eval_perf': float(eval_perf_mean),
                            }

                    if hp.get('inclusive_loss_stop', False):
                        loss_stop_reached = eval_loss_mean <= hp['loss_thr']
                    else:
                        # Preserve the historical joint-task stopping rule exactly.
                        loss_stop_reached = eval_loss_mean < hp['loss_thr']
                    if loss_stop_reached and eval_perf_mean >= hp['perf_thr']:
                        break

                if hp['save_every'] > 0 and (tr + 1) % hp['save_every'] == 0:
                    save_checkpoint(last_ckpt_path, rnn, optimizer=optimizer, step=tr + 1,
                                    best_metric=float(best_eval_loss), hp=hp)

            elapsed_time = time.time() - start_time
            print("elapsed_time=", elapsed_time, "s")
            save_checkpoint(last_ckpt_path, rnn, optimizer=optimizer, step=tr + 1,
                            best_metric=float(best_eval_loss), hp=hp)
        elif Mode == 'eval':
            print('EVALUATION STARTED ...')
            ckpt_to_load = hp['eval_ckpt_path'] if hp['eval_ckpt_path'] else hp['resume_path']
            if not ckpt_to_load:
                if os.path.exists(eval_loss_ckpt_path):
                    ckpt_to_load = eval_loss_ckpt_path
                else:
                    for legacy_ckpt_path in legacy_eval_loss_ckpt_paths:
                        if os.path.exists(legacy_ckpt_path):
                            ckpt_to_load = legacy_ckpt_path
                            print('Using legacy speed-coded checkpoint for tagged eval: {}'.format(legacy_ckpt_path))
                            break
                    if not ckpt_to_load:
                        print('Skipped eval for {}: missing required checkpoint {}'.format(
                            run_id, os.path.basename(eval_loss_ckpt_path)))
                        continue
            load_checkpoint(ckpt_to_load, rnn, optimizer=None, load_optimizer=False,
                            strict=hp['strict_load'])
            print('Loaded checkpoint for eval: {}'.format(ckpt_to_load))
            losses = np.zeros((1,))
            perfEvals = np.zeros((1,))
            lossEvals = np.zeros((1,))
            tr = 0
            eval_loss_mean, eval_perf_mean, eval_losses, eval_perfs, eval_os, eval_o_nets, eval_us, \
                eval_zs, eval_rs, eval_ms, stim, target, target_mask = run_evaluation()
            perfEvals[0] = eval_perf_mean
            lossEvals[0] = eval_loss_mean
            print("Eval Loss: %.4f, Perf: %.4f" % (eval_loss_mean, eval_perf_mean))
            reached_loss_checkpoints = {}

        train_loss_stop = int(tr + 1) if Mode == 'train' else int(len(losses))
        train_eval_stop = int(evalCount) if Mode == 'train' else int(len(lossEvals))

        save_train_eval_dynamics = bool(hp.get('save_train_eval_dynamics', False))
        if Mode == 'train':
            if save_train_eval_dynamics:
                _, _, eval_losses, eval_perfs, eval_os, eval_o_nets, eval_us, eval_zs, eval_rs, eval_ms, stim, target, target_mask = run_evaluation()
            else:
                eval_losses = lossEvals[0:train_eval_stop]
                eval_perfs = perfEvals[0:train_eval_stop]
                eval_os, eval_o_nets, eval_us, eval_zs, eval_rs, eval_ms = (None, None, None, None, None, None)

        run_stub = '{}_train'.format(run_id) if Mode == 'train' else run_id
        if Mode == 'eval':
            run_stub = '{}_eval_{}{}'.format(
                run_id,
                hp.get('eval_grid_name', 'grid'),
                eval_rates_file_suffix(hp),
            )
        result_out_dir = result_output_dir(out_dir, hp)
        os.makedirs(result_out_dir, exist_ok=True)
        recurrent_w_np = rnn.recurrent_weight().detach().cpu().numpy()
        m_vectors_np = None
        n_vectors_np = None
        if getattr(rnn, 'rnn_type', 'full_rank') == 'low_rank':
            m_vectors_np = rnn.m_vectors.detach().cpu().numpy()
            n_vectors_np = rnn.n_vectors.detach().cpu().numpy()

        condition_types = [str(c.get('condition_type', 'unknown')) for c in eval_conditions]
        size_levels_eval = np.asarray(
            [float(c.get('size_level', np.nan)) for c in eval_conditions], dtype=np.float32)
        speed_levels_eval = np.asarray(
            [float(c.get('speed_level', np.nan)) for c in eval_conditions], dtype=np.float32)
        target_radii_eval = np.asarray(
            [float(c.get('target_radius', np.nan)) for c in eval_conditions], dtype=np.float32)
        target_durations_eval = np.asarray(
            [float(c.get('target_duration', np.nan)) for c in eval_conditions], dtype=np.float32)

        eval_block = {
            'conditions': eval_conditions,
            'condition_types': condition_types,
            'size_levels': size_levels_eval,
            'speed_levels': speed_levels_eval,
            'target_radii': target_radii_eval,
            'target_durations': target_durations_eval,
            'performance': eval_perfs,
            'losses': eval_losses,
            'eval_grid_name': hp.get('eval_grid_name', ''),
            'dynamics_saved': bool(eval_os is not None),
            'rates_saved': bool(eval_rs is not None and hp.get('save_eval_rates', False)),
            'outputs_net_saved': bool(eval_o_nets is not None and hp.get('save_eval_outputs_net', False)),
        }
        if eval_os is not None:
            eval_block['inputs'] = eval_us
            eval_block['targets'] = eval_zs
            eval_block['masks'] = eval_ms
            eval_block['outputs'] = eval_os
            if hp.get('save_eval_outputs_net', False):
                eval_block['outputs_net'] = eval_o_nets
            if hp.get('save_eval_rates', False):
                eval_block['rates'] = eval_rs
        if Mode == 'eval':
            eval_block['n_conditions'] = int(len(eval_conditions))
            eval_block['trials_per_condition'] = int(hp['eval_tr'])
            eval_block['eval_layout'] = 'condition_trial_out_time'
        else:
            eval_block['eval_layout'] = 'step_out_batch_time'
            eval_block['train_eval_batches'] = int(hp['train_eval_batches'])
        result_payload = {
            'mode': Mode,
            'hp': hp,
            'train': {
                'final_trial': int(tr),
                'losses': losses[0:train_loss_stop],
                'perfEvals': perfEvals[0:train_eval_stop],
                'lossEvals': lossEvals[0:train_eval_stop],
                'lossCheckpointThresholds': list(loss_checkpoint_values),
                'lossCheckpointFirstHits': reached_loss_checkpoints,
            },
            'eval': eval_block,
            'snapshot': {
                'stim': stim,
                'target': target,
                'target_mask': target_mask,
            },
            'model': {
                'w0': rnn.w0,
                'w': recurrent_w_np,
                'm': rnn.m.detach().cpu().numpy(),
                'w_in0': rnn.w_in0,
                'w_in': rnn.w_in.detach().cpu().numpy(),
                'w_out0': rnn.w_out0,
                'w_out': rnn.w_out.detach().cpu().numpy(),
                'b_rec0': rnn.b_rec0,
                'b_rec': rnn.b_rec.detach().cpu().numpy(),
                'b_out0': rnn.b_out0,
                'b_out': rnn.b_out.detach().cpu().numpy(),
                'N': rnn.N,
                'exc': rnn.exc,
                'inh': rnn.inh,
                'rnn_type': rnn.rnn_type,
                'rank': rnn.rank,
                'm_vectors': m_vectors_np,
                'n_vectors': n_vectors_np,
            },
        }
        torch.save(result_payload, os.path.join(result_out_dir, run_stub + '.pt'))

        if hp['export_mat']:
            var = {}
            hp_for_mat = dict(hp)
            if hp_for_mat.get('rank') is None:
                hp_for_mat['rank'] = -1
            var['hp'] = hp_for_mat
            for _meta_key in ('eval_layout', 'n_conditions', 'trials_per_condition',
                              'train_eval_batches', 'dynamics_saved', 'rates_saved',
                              'outputs_net_saved'):
                if _meta_key in eval_block:
                    var[_meta_key] = eval_block[_meta_key]
            var['eval_grid_name'] = hp.get('eval_grid_name', '')
            var['eval_size_levels'] = size_levels_eval
            var['eval_speed_levels'] = speed_levels_eval
            var['eval_target_radii'] = target_radii_eval
            var['eval_target_durations'] = target_durations_eval
            var['eval_perfs'] = eval_block['performance']
            var['eval_losses'] = eval_block['losses']
            var['eval_condition_types'] = np.asarray(condition_types, dtype=object)
            var['w0'] = rnn.w0
            var['b_out0'] = rnn.b_out0
            var['b_rec0'] = rnn.b_rec0
            var['stim'] = stim
            var['w'] = recurrent_w_np
            var['target'] = target
            var['target_mask'] = target_mask
            var['w_out'] = rnn.w_out.detach().cpu().numpy()
            var['w_out0'] = rnn.w_out0
            var['m'] = rnn.m.detach().cpu().numpy()
            var['N'] = rnn.N
            var['exc'] = rnn.exc
            var['inh'] = rnn.inh
            var['w_in'] = rnn.w_in.detach().cpu().numpy()
            var['w_in0'] = rnn.w_in0
            var['b_out'] = rnn.b_out.detach().cpu().numpy()
            var['b_rec'] = rnn.b_rec.detach().cpu().numpy()
            var['rnn_type'] = hp['rnn_type']
            var['rank'] = -1 if hp['rank'] is None else int(hp['rank'])
            if m_vectors_np is not None:
                var['m_vectors'] = m_vectors_np
                var['n_vectors'] = n_vectors_np
            var['losses'] = losses[0:train_loss_stop]
            var['tr'] = tr
            if 'inputs' in eval_block:
                var['eval_us'] = eval_block['inputs']
                var['eval_zs'] = eval_block['targets']
                var['eval_ms'] = eval_block['masks']
                var['eval_os'] = eval_block['outputs']
            if 'outputs_net' in eval_block:
                var['eval_o_nets'] = eval_block['outputs_net']
            if 'rates' in eval_block:
                var['eval_rs'] = eval_block['rates']
            var['perfEvals'] = perfEvals[0:train_eval_stop]
            var['lossEvals'] = lossEvals[0:train_eval_stop]
            scipy.io.savemat(os.path.join(result_out_dir, run_stub + '.mat'), var, long_field_names=True)



