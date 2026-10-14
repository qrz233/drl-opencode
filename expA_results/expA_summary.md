# Experiment A - Block A dual diagnostics summary


## replay

| metric | value |
|---|---|
| fidelity_pass | 1.0 |
| qlen_max_absdiff | 0.0 |
| chcapa_max_absdiff | 0.0 |

## equivalence

| metric | value |
|---|---|
| max_abs_diff_D | 0.0 |
| status_mismatch_slots | 0.0 |

## base_solve

| metric | value |
|---|---|
| n_slots | 501.0 |
| status=optimal | 500.0 |
| status=optimal_inaccurate | 1.0 |
| solver=CLARABEL | 500.0 |
| solver=SCS | 1.0 |
| max_access_resid_Mb | 0.057301 |
| max_backhaul_resid_Mb | 1e-06 |
| max_demand_resid_Mb | 0.0 |
| max_complSlack_access | 0.0 |
| max_complSlack_backhaul | 4e-06 |

## dual_sanity

| metric | value |
|---|---|
| min_dual_overall | -0.0 |
| n_slots_dual_negative | 0.0 |

## backhaul

| metric | value |
|---|---|
| n_eval_pairs | 2004.0 |
| pct_tight | 41.716567 |
| pct_tight_nu_positive | 100.0 |
| pct_tight_nu_zero_degenerate | 0.0 |
| pct_slack_nu_zero_expected | 99.657534 |
| pct_slack_nu_positive_violation | 0.342466 |
| mean_utilisation_S_over_C | 0.939177 |
| degenerate_frac_demand_capped | nan |
| degenerate_mean_pool_bh_dual | nan |

## fd_backhaul

| metric | value |
|---|---|
| n_fd | 2004.0 |
| rho_all | 0.856879 |
| p_all | 0.0 |
| rho_tight | 0.988879 |
| p_tight | 0.0 |
| med_abs_err_tight | 4.1e-05 |
| med_rel_err_strong | 0.000912 |
| pct_within10pct | 97.84689 |
| m_high_nu | 0.483985 |
| m_low_nu | -0.0 |
| n_stable_pairs | 2004.0 |
| pct_unstable_rel>0.5 | 12.924152 |

## backhaul

| metric | value |
|---|---|
| n_slack_nu_positive | 4.0 |
| slack_nu_pos_eta_median | 0.001525 |
| slack_nu_pos_eta_max | 0.003493 |
| slack_nu_pos_nu_max | 0.0 |

## pools

| metric | value |
|---|---|
| max_pool_access_dual | 3e-06 |
| pct_slots_pool_access_binding | 21.556886 |
| max_pool_backhaul_dual | 18.542054 |
| pct_slots_pool_backhaul_binding | 44.311377 |

## access

| metric | value |
|---|---|
| n_eval_users | 49957.0 |
| pct_tight | 0.002002 |
| pct_tight_mu_positive | 0.0 |
| pct_tight_mu_zero_degenerate | 100.0 |
| pct_loose | 97.315691 |
| pct_loose_mu_positive_violation | 2.353135 |
| max_mu_on_loose_users | 0.0 |
| pct_active_demand_capped | 54.894111 |

## fd_access

| metric | value |
|---|---|
| n_fd_tight | 0.0 |
| n_fd_loose | 400.0 |
| mean_mu_loose | 0.0 |
| max_mu_loose | 0.0 |
| mean_m_loose | 0.0 |
| max_abs_m_loose | 0.000159 |
| rho_loose | 0.16563 |
| p_loose | 0.000883 |

## anomalies

| metric | value |
|---|---|
| solver-fallback | 1.0 |
| fd-ac-solver-fallback | 1.0 |
| fd-bh-solver-fallback | 1.0 |
