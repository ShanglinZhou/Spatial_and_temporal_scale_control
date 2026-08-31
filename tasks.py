"""Analytic circle motor trajectory tasks for the RNN sweep.

The three supported tasks share the same input, target, and saved-data schema:

- ``motorTraj_circle`` jointly varies circle radius and movement duration.
- ``motorTraj_circle_time`` varies duration and fixes radius at the center cue.
- ``motorTraj_circle_space`` varies radius and fixes duration at the center cue.

The public functions keep the same names used by ``PARAM.py`` and recurrent
analysis scripts, so changing ``hp['task']`` is sufficient to switch tasks.

Input convention
----------------
- ``num_In == 3``.
- Channel 0 is a go cue during ``stim_on : stim_on + stim_dur``.
- Channels 1 and 2 are radius and time/speed cue amplitudes. They are zero
  before movement onset and tonic afterwards.

Target convention
-----------------
- ``num_Out == 2``.
- Each condition is a clockwise circle through the common point ``(0, 1)``.
- Radius cue levels map to geometric radii.
- Time cue levels map to movement durations. When
  ``use_speed_coded_time_input`` is true, larger input levels map to shorter
  target durations.
"""

import numpy as np


MOTOR_SPACE_TIME_INPUT_LEVELS = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
MOTOR_CIRCLE_SPEED_DURATIONS_S = np.array([0.7, 1.3, 1.9, 2.5, 3.1], dtype=np.float32)
MOTOR_CIRCLE_RADII = np.array([0.45, 0.725, 1.0, 1.275, 1.55], dtype=np.float32)
MOTOR_CIRCLE_TRAJ_START_XY = np.array([0.0, 1.0], dtype=np.float32)
MOTOR_TRAJ_CIRCLE_TASKS = frozenset({
    'motorTraj_circle',
    'motorTraj_circle_time',
    'motorTraj_circle_space',
})


def map_level_to_value(level, base_levels, base_values, allow_extrapolate=False):
    """Map a cue level to a value by linear interpolation."""
    level = float(level)
    base_levels = np.asarray(base_levels, dtype=np.float64)
    base_values = np.asarray(base_values, dtype=np.float64)
    if base_levels.ndim != 1 or base_values.ndim != 1 or base_levels.size != base_values.size:
        raise ValueError('base_levels and base_values must be 1D arrays with the same length')
    if base_levels.size < 2:
        raise ValueError('At least two base levels are required')

    order = np.argsort(base_levels)
    xs = base_levels[order]
    ys = base_values[order]

    if level < xs[0]:
        if not allow_extrapolate:
            raise ValueError('level {} is below interpolation range [{}, {}]'.format(level, xs[0], xs[-1]))
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + slope * (level - xs[0]))
    if level > xs[-1]:
        if not allow_extrapolate:
            raise ValueError('level {} is above interpolation range [{}, {}]'.format(level, xs[0], xs[-1]))
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + slope * (level - xs[-1]))

    return float(np.interp(level, xs, ys))


def circle_speed_level_to_duration(level, allow_extrapolate=False):
    """Map a circle cue level to movement duration in seconds.

    This preserves the original helper behavior. Use
    ``circle_time_input_level_to_duration(..., speed_coded=True)`` when larger
    cue amplitudes should mean faster movements and shorter durations.
    """
    return map_level_to_value(
        level,
        MOTOR_SPACE_TIME_INPUT_LEVELS,
        MOTOR_CIRCLE_SPEED_DURATIONS_S,
        allow_extrapolate=allow_extrapolate,
    )


def circle_time_input_level_to_duration(level, allow_extrapolate=False, speed_coded=False, base_levels=None):
    """Map a circle time/speed cue level to movement duration in seconds."""
    durations = MOTOR_CIRCLE_SPEED_DURATIONS_S[::-1] if speed_coded else MOTOR_CIRCLE_SPEED_DURATIONS_S
    if base_levels is None:
        base_levels = MOTOR_SPACE_TIME_INPUT_LEVELS
    return map_level_to_value(
        level,
        base_levels,
        durations,
        allow_extrapolate=allow_extrapolate,
    )


def circle_size_level_to_radius(level, allow_extrapolate=False, base_levels=None):
    """Map a circle size cue level to geometric radius."""
    if base_levels is None:
        base_levels = MOTOR_SPACE_TIME_INPUT_LEVELS
    return map_level_to_value(
        level,
        base_levels,
        MOTOR_CIRCLE_RADII,
        allow_extrapolate=allow_extrapolate,
    )


def _levels_from_hp(hp, key, fallback):
    if key in hp and hp[key] is not None:
        return np.asarray(hp[key], dtype=np.float32)
    return np.asarray(fallback, dtype=np.float32)


def _train_input_levels(hp):
    return _levels_from_hp(hp, 'input_set', MOTOR_SPACE_TIME_INPUT_LEVELS)


def _circle_task(hp):
    task = str(hp.get('task', ''))
    if task not in MOTOR_TRAJ_CIRCLE_TASKS:
        raise ValueError(
            'Unsupported circle task {!r}; expected one of {}'.format(
                task, sorted(MOTOR_TRAJ_CIRCLE_TASKS)
            )
        )
    return task


def _center_input_level(hp):
    """Return the center of the task's trained cue levels."""
    levels = np.sort(np.unique(_train_input_levels(hp).astype(np.float64)))
    if levels.size == 0:
        raise ValueError('At least one training input level is required')
    return float(np.median(levels))


def _fixed_level(hp, axis):
    key = 'circle_fixed_{}_level'.format(axis)
    return float(hp[key]) if hp.get(key) is not None else _center_input_level(hp)


def _apply_task_fixed_levels(hp, size_level, speed_level):
    """Clamp the non-controlled cue to its center value for single-axis tasks."""
    task = _circle_task(hp)
    if task == 'motorTraj_circle_time':
        size_level = _fixed_level(hp, 'size')
    elif task == 'motorTraj_circle_space':
        speed_level = _fixed_level(hp, 'time')
    return float(size_level), float(speed_level)


def _circle_target_map_levels(hp):
    return _levels_from_hp(hp, 'circle_target_map_levels', MOTOR_SPACE_TIME_INPUT_LEVELS)


def _combined_eval_levels(hp):
    if 'eval_size_levels' in hp and hp['eval_size_levels'] is not None:
        size_levels = np.asarray(hp['eval_size_levels'], dtype=np.float32)
    else:
        parts = [
            _levels_from_hp(hp, 'eval_extra_low_levels', []),
            _levels_from_hp(hp, 'eval_train_levels', MOTOR_SPACE_TIME_INPUT_LEVELS),
            _levels_from_hp(hp, 'eval_interp_levels', []),
            _levels_from_hp(hp, 'eval_extra_high_levels', []),
        ]
        size_levels = np.unique(np.concatenate([p for p in parts if p.size > 0])).astype(np.float32)

    if 'eval_speed_levels' in hp and hp['eval_speed_levels'] is not None:
        speed_levels = np.asarray(hp['eval_speed_levels'], dtype=np.float32)
    else:
        parts = [
            _levels_from_hp(hp, 'eval_extra_low_levels', []),
            _levels_from_hp(hp, 'eval_train_levels', MOTOR_SPACE_TIME_INPUT_LEVELS),
            _levels_from_hp(hp, 'eval_interp_levels', []),
            _levels_from_hp(hp, 'eval_extra_high_levels', []),
        ]
        speed_levels = np.unique(np.concatenate([p for p in parts if p.size > 0])).astype(np.float32)

    return np.sort(size_levels), np.sort(speed_levels)


def classify_eval_condition(size_level, speed_level, train_levels=None):
    if train_levels is None:
        train_levels = MOTOR_SPACE_TIME_INPUT_LEVELS
    train_levels = np.asarray(train_levels, dtype=np.float32)
    size_level = float(size_level)
    speed_level = float(speed_level)
    lo = float(np.min(train_levels))
    hi = float(np.max(train_levels))
    size_train = np.any(np.isclose(train_levels, size_level, atol=1e-5))
    speed_train = np.any(np.isclose(train_levels, speed_level, atol=1e-5))
    if size_train and speed_train:
        return 'train'
    if lo <= size_level <= hi and lo <= speed_level <= hi:
        return 'interpolation'
    return 'extrapolation'


def list_eval_conditions(hp):
    """Return the structured evaluation grid for the selected circle task."""
    task = _circle_task(hp)

    train_levels = _levels_from_hp(hp, 'eval_train_levels', MOTOR_SPACE_TIME_INPUT_LEVELS)
    use_flexible_eval = any(k in hp for k in (
        'eval_size_levels',
        'eval_speed_levels',
        'eval_train_levels',
        'eval_interp_levels',
        'eval_extra_low_levels',
        'eval_extra_high_levels',
    ))
    if use_flexible_eval:
        size_levels, speed_levels = _combined_eval_levels(hp)
    else:
        size_levels = MOTOR_SPACE_TIME_INPUT_LEVELS
        speed_levels = MOTOR_SPACE_TIME_INPUT_LEVELS

    if task == 'motorTraj_circle_time':
        size_levels = np.asarray([_fixed_level(hp, 'size')], dtype=np.float32)
    elif task == 'motorTraj_circle_space':
        speed_levels = np.asarray([_fixed_level(hp, 'time')], dtype=np.float32)

    allow_extrapolate = bool(hp.get('eval_allow_extrapolate', False))
    target_map_levels = _circle_target_map_levels(hp)
    speed_coded = bool(hp.get('use_speed_coded_time_input', False))

    rows = []
    for size_level in size_levels:
        for speed_level in speed_levels:
            rows.append({
                'size_level': float(size_level),
                'speed_level': float(speed_level),
                'condition_type': classify_eval_condition(size_level, speed_level, train_levels),
                'target_radius': circle_size_level_to_radius(
                    size_level,
                    allow_extrapolate=allow_extrapolate,
                    base_levels=target_map_levels,
                ),
                'target_duration': circle_time_input_level_to_duration(
                    speed_level,
                    allow_extrapolate=allow_extrapolate,
                    speed_coded=speed_coded,
                    base_levels=target_map_levels,
                ),
            })
    return rows


def _check_num_in(hp):
    _circle_task(hp)
    if hp['num_In'] != 3:
        raise ValueError('Circle tasks expect hp["num_In"] == 3, got {}'.format(hp['num_In']))


def _set_go_and_motor_context(input_arr, trial_i, stim_on, stim_dur, size_level, speed_level):
    s_on = int(np.asarray(stim_on).reshape(-1)[0])
    s_dur = int(np.asarray(stim_dur).reshape(-1)[0])
    t_move = s_on + s_dur
    input_arr[0, trial_i, s_on:s_on + s_dur] = 1.0
    input_arr[1, trial_i, :t_move] = 0.0
    input_arr[2, trial_i, :t_move] = 0.0
    if t_move < input_arr.shape[2]:
        input_arr[1, trial_i, t_move:] = float(size_level)
        input_arr[2, trial_i, t_move:] = float(speed_level)


def _build_target_circle(hp, label, batch_size):
    if batch_size is None:
        batch_size = hp['batch_size']

    num_out = hp['num_Out']
    if num_out != 2:
        raise ValueError('Circle tasks expect hp["num_Out"] == 2, got {}'.format(num_out))

    T = int(hp['T'])
    stim_on = int(np.asarray(hp['stim_on']).reshape(-1)[0])
    stim_dur = int(np.asarray(hp['stim_dur']).reshape(-1)[0])
    dt = float(hp['dt'])

    target = np.zeros((num_out, batch_size, T), dtype=np.float32)
    target_mask = np.zeros((num_out, batch_size, T), dtype=np.float32)

    for i in range(batch_size):
        row = label[i]
        radius = float(row['radius'])
        duration_s = float(row['duration_s'])
        interval = int(np.round(duration_s * 1000.0 / dt))
        interval = max(1, min(interval, T - (stim_on + stim_dur)))

        t_frac = np.linspace(0.0, 1.0, interval, dtype=np.float64)
        theta = np.pi / 2.0 - 2.0 * np.pi * t_frac
        cy = 1.0 - radius
        seg = np.stack(
            (radius * np.cos(theta), cy + radius * np.sin(theta)),
            axis=0,
        ).astype(np.float32)

        t0 = stim_on + stim_dur
        target[:, i, stim_on:t0] = MOTOR_CIRCLE_TRAJ_START_XY[:, np.newaxis]
        target[:, i, t0:t0 + interval] = seg
        target_mask[:, i, :t0 + interval] = 1.0

    return np.float32(target), np.float32(target_mask)


def generate_input_motorTraj_circle(hp, mode='train', eval_conditions=None, batch_size=None):
    _check_num_in(hp)
    if batch_size is None:
        batch_size = hp['batch_size']

    input_arr = np.zeros((hp['num_In'], batch_size, hp['T']), dtype=np.float32)
    levels = _train_input_levels(hp)
    target_map_levels = _circle_target_map_levels(hp)
    allow_extrapolate = bool(hp.get('eval_allow_extrapolate', False))
    speed_coded = bool(hp.get('use_speed_coded_time_input', False))
    label = []

    for i in range(batch_size):
        if mode == 'eval' and eval_conditions is not None:
            size_level = float(eval_conditions['size_level'])
            speed_level = float(eval_conditions['speed_level'])
        else:
            size_level = float(levels[np.random.randint(0, len(levels))])
            speed_level = float(levels[np.random.randint(0, len(levels))])

        size_level, speed_level = _apply_task_fixed_levels(hp, size_level, speed_level)

        duration_s = circle_time_input_level_to_duration(
            speed_level,
            allow_extrapolate=allow_extrapolate,
            speed_coded=speed_coded,
            base_levels=target_map_levels,
        )
        radius = circle_size_level_to_radius(
            size_level,
            allow_extrapolate=allow_extrapolate,
            base_levels=target_map_levels,
        )

        _set_go_and_motor_context(input_arr, i, hp['stim_on'], hp['stim_dur'], size_level, speed_level)
        label.append({
            'size_level': size_level,
            'speed_level': speed_level,
            'duration_s': duration_s,
            'target_duration': duration_s,
            'radius': radius,
            'target_radius': radius,
        })

    return input_arr, label


def generate_target_motorTraj_circle(hp, label, batch_size=None):
    return _build_target_circle(hp, label, batch_size)


# Explicit aliases make the two task names discoverable while preserving the
# shared API used by PARAM.py and downstream analysis code.
def generate_input_motorTraj_circle_time(hp, mode='train', eval_conditions=None, batch_size=None):
    return generate_input_motorTraj_circle(
        hp, mode=mode, eval_conditions=eval_conditions, batch_size=batch_size
    )


def generate_target_motorTraj_circle_time(hp, label, batch_size=None):
    return generate_target_motorTraj_circle(hp, label, batch_size=batch_size)


def generate_input_motorTraj_circle_space(hp, mode='train', eval_conditions=None, batch_size=None):
    return generate_input_motorTraj_circle(
        hp, mode=mode, eval_conditions=eval_conditions, batch_size=batch_size
    )


def generate_target_motorTraj_circle_space(hp, label, batch_size=None):
    return generate_target_motorTraj_circle(hp, label, batch_size=batch_size)
