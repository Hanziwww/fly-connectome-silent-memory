"""One vocabulary for behaviour outcomes, training outcomes and interventions across all sources."""

# Behavioural end states of an autonomous rollout.
BEHAVIOR = {
    'solved': 'solved',
    'repeated_state': 'repeated_state',
    'proven_corner_deadlock': 'deadlock',
    'blocked': 'blocked',
    'action_limit': 'action_limit',
    'wrong_irreversible_push': 'irreversible_failure',   # remote batch 1 label
    'irreversible_deadlock': 'irreversible_failure',     # remote supplement label
    'running': 'incomplete',
}

# Training end states (study-specific labels kept; grouped for summaries).
TRAINING = {
    'learned_solved': ('acquired', 'solved after training'),
    'original_solved': ('acquired', 'formal lineage run solved its original task'),
    'already_solved': ('existing', 'solved before any update'),
    'stagnation': ('stopped_rule', 'segment ended without qualifying progress'),
    'segment_limit': ('stopped_rule', 'third time segment exhausted'),
    'frozen_baseline': ('baseline', 'no trainable parameters; evaluation only'),
    'untrained_baseline': ('baseline', 'evaluation of the untrained configuration'),
    'stopped': ('stopped_user', 'stopped by the study STOP file'),
    'numerical_error': ('failed', 'numerical failure (recomputation tolerance)'),
    'implementation_error': ('failed', 'implementation failure'),
    'passed': ('check', 'acceptance or import check passed'),
    'completed': ('evaluation', 'evaluation job completed'),
    'complete': ('evaluation', 'mechanism job completed'),
    'verified': ('check', 'independent reload verified'),
    'pending': ('never_run', 'registered but never executed'),
}


def behavior(status):
    if status is None:
        return None
    return BEHAVIOR.get(status, status)


def training_group(status):
    return TRAINING.get(status, ('other', ''))[0]


# Intervention taxonomy: (source, raw label) -> canonical label, plus attributes.
# Canonical labels: baseline, sham, restore, same_cue_swap, opposite_cue_swap, neural_clear,
# full_reset, reset_each_window, clear_<component>, swap_<component>, reset_resources,
# interpolate_cue (dose), norm_matched_scramble (dose), group_suppression, readout_replacement.
QA_CHECKS = {'sham', 'restore', 'same_cue_swap'}

R0_KIND = {
    'voltage': 'clear_voltage', 'current': 'clear_current', 'adaptation': 'clear_adaptation',
    'delay': 'clear_delay_buffer', 'all': 'full_reset', 'swap': 'opposite_cue_swap', 'restore': 'restore',
}

RS_KIND = {
    None: 'baseline', 'none': 'baseline', 'sham': 'sham', 'same_cue': 'same_cue_swap', 'swap': 'opposite_cue_swap',
    'neural_clear': 'neural_clear', 'voltage': 'clear_voltage', 'current': 'clear_current', 'restore': 'restore',
    'reset_each_window': 'reset_each_window', 'adaptation': 'clear_adaptation', 'delay': 'clear_delay_buffer',
    'all': 'neural_clear',
}

COMPONENT = {'voltage': 'voltage', 'current': 'current', 'adaptation': 'adaptation', 'adapt': 'adaptation',
             'resources': 'resources', 'res': 'resources', 'delay': 'delay_buffer', 'buffer': 'delay_buffer',
             'all': 'all_state', 'continuous': 'continuous_combined'}
