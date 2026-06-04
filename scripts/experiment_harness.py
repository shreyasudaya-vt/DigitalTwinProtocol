#!/usr/bin/env python3
"""Experiment harness: run PDR sweep by driving Anasta edge/twin classes in-process
Produces aggregated plots into python_reports/experiment_summary.png and CSV.
"""
import os
import time
import random
import struct
import numpy as np
import matplotlib.pyplot as plt

from digital_twin import DigitalTwinServer
from edge_node import AnastaEdgeNode

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPORT_DIR = os.path.join(ROOT, 'python_reports')
os.makedirs(REPORT_DIR, exist_ok=True)

def make_header(seq, tier, telemetry=22.4):
    ts = int(time.time()) & 0xFFFFFFFF
    return struct.pack(
        ">IIBfB",
        ts,
        seq,
        int(tier),
        float(telemetry),
        0,
    )

def run_pdr_sweep(pdr_values, packets_per_point=200, run_offset=0):
    summary = []
    for pdr in pdr_values:
        print(f"\n=== Running PDR={pdr:.2f} ({packets_per_point} packets) ===")

        # instantiate fresh components per sweep for clean enrollment
        # Use unique ports per run to avoid address-in-use collisions
        base_port = 5000 + int(pdr * 100) + int(run_offset * 10)
        feedback_port = 9001 + int(pdr * 100) + int(run_offset * 10)
        twin = DigitalTwinServer(listen_port=base_port, feedback_port=feedback_port)
        node = AnastaEdgeNode(ns3_port=base_port, feedback_port=feedback_port)

        # calibrate PCA on edge (may fallback to synthetic if data missing)
        node.calibrate_from_hardware()

        # Prepare a stable base vector (use first available sweep or random)
        dev_sweeps = node.device_files_map.get(node.target_device, {})
        if len(dev_sweeps) > 0:
            idx = sorted(dev_sweeps.keys())[0]
            from final_pca import load_sweep_vector, REF_FREQ, USE_PHASE
            stable_base = load_sweep_vector(dev_sweeps[idx], ref_freq=REF_FREQ, use_phase=USE_PHASE)
        else:
            stable_base = np.random.randn(len(node.pca.components_[0]))

        # ensure twin starts clean
        successes = 0
        total_hd = 0
        alarms = 0
        delivered = 0

        for seq in range(packets_per_point):
            # set tier according to PDR (mimic feedback loop)
            tier = 2 if pdr < 0.80 else 1

            # create a (deterministic) sweep variation around stable base
            noise = np.random.normal(0, 0.001, stable_base.shape)
            real_sweep = stable_base + noise

            k_auth, k_health = node.transform_hardware_sweep(real_sweep)
            n_i, s_i = node._generate_crypto_masks(seq)

            header = make_header(seq, tier)

            if tier == 1:
                p_auth = bytes(a ^ b for a, b in zip(k_auth, n_i))
                p_health = struct.pack(">32f", *list(k_health[:32]))
                packet = header + p_auth + p_health
            else:
                k_auth_comp = node._fountain_compress(k_auth, s_i)
                n_i_comp = node._fountain_compress(n_i, s_i)
                p_fountain = bytes(a ^ b for a, b in zip(k_auth_comp, n_i_comp))
                packet = header + p_fountain

            # Simulate delivery according to PDR
            if random.random() <= pdr:
                delivered += 1
                twin.process_packet(packet)
            else:
                # packet dropped
                pass

            # small sleep to avoid overwhelming in-process objects
            time.sleep(0.005)

        # wait a short moment for twin to flush
        time.sleep(0.5)

        # parse twin log CSV for results
        csv_path = "thesis_telemetry_log.csv"
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            # select rows for this run (last packets_per_point entries)
            df_run = df.tail(packets_per_point)
            mean_hd = df_run['Hamming_Distance'].mean()
            alarm_rate = df_run['Threshold_Alarm'].mean()
            # define success as non-alarm and HD <= 5
            success_rate = ((df_run['Threshold_Alarm'] == 0) & (df_run['Hamming_Distance'] <= 5)).mean()
        else:
            mean_hd = float('nan')
            alarm_rate = float('nan')
            success_rate = float('nan')

        print(f"PDR {pdr:.2f}: delivered {delivered}/{packets_per_point} | success_rate {success_rate:.3f} | mean_hd {mean_hd:.2f} | alarm_rate {alarm_rate:.3f}")

        summary.append({
            'pdr': pdr,
            'delivered': delivered,
            'packets': packets_per_point,
            'success_rate': success_rate,
            'mean_hd': mean_hd,
            'alarm_rate': alarm_rate
        })

        # cleanup sockets and resources
        try:
            twin.log_file.close()
            twin.rx_sock.close()
            twin.tx_sock.close()
        except Exception:
            pass
        try:
            node.close()
        except Exception:
            pass

    return summary

def plot_summary(summary, outpath_png, outpath_csv):
    import pandas as pd
    df = pd.DataFrame(summary)
    df.to_csv(outpath_csv, index=False)

    plt.figure(figsize=(6,4))
    plt.plot(df['pdr'], df['success_rate'], '-o', label='Authentication Success')
    plt.plot(df['pdr'], df['alarm_rate'], '-s', label='Alarm Rate')
    plt.xlabel('PDR')
    plt.ylabel('Rate')
    plt.title('Authentication Success & Alarm Rate vs PDR')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=300)
    print(f"Saved summary plots to {outpath_png} and CSV to {outpath_csv}")

if __name__ == '__main__':
    # Default coarse sweep (can be overridden with env vars)
    import os
    pdr_env = os.environ.get('EXP_PDRS')
    pkt_env = os.environ.get('EXP_PACKETS')

    if pdr_env:
        pdr_values = [float(x) for x in pdr_env.split(',')]
    else:
        pdr_values = [0.2, 0.4, 0.6, 0.8, 1.0]

    packets_per_point = int(pkt_env) if pkt_env else 120

    repeats = int(os.environ.get('EXP_REPEATS', '1'))

    if repeats <= 1:
        summary = run_pdr_sweep(pdr_values, packets_per_point=packets_per_point)
        out_png = os.path.join(REPORT_DIR, 'experiment_summary.png')
        out_csv = os.path.join(REPORT_DIR, 'experiment_summary.csv')
        plot_summary(summary, out_png, out_csv)
    else:
        # run multiple repeats and aggregate per-PDR statistics
        import pandas as pd
        all_runs = []
        for r in range(repeats):
            print(f"\n*** Repeat {r+1}/{repeats} ***")
            s = run_pdr_sweep(pdr_values, packets_per_point=packets_per_point, run_offset=r)
            all_runs.append(s)

        # aggregate
        agg_rows = []
        for i, pdr in enumerate(pdr_values):
            vals_success = [run[i]['success_rate'] for run in all_runs]
            vals_alarm = [run[i]['alarm_rate'] for run in all_runs]
            vals_meanhd = [run[i]['mean_hd'] for run in all_runs]

            # convert NaNs to np.nan and compute mean and 95% CI with nanmean
            arr_s = np.array([float('nan') if v is None or v=='' else float(v) for v in vals_success], dtype=float)
            arr_a = np.array([float('nan') if v is None or v=='' else float(v) for v in vals_alarm], dtype=float)
            arr_h = np.array([float('nan') if v is None or v=='' else float(v) for v in vals_meanhd], dtype=float)

            def ci95(x):
                x = x[~np.isnan(x)]
                if len(x) == 0:
                    return (np.nan, np.nan, np.nan)
                m = x.mean()
                se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
                lo = m - 1.96 * se
                hi = m + 1.96 * se
                return (m, lo, hi)

            ms, ls, hs = ci95(arr_s)
            ma, la, ha = ci95(arr_a)
            mh, lh, hh = ci95(arr_h)

            agg_rows.append({
                'pdr': pdr,
                'mean_success': ms, 'ci_success_lo': ls, 'ci_success_hi': hs,
                'mean_alarm': ma, 'ci_alarm_lo': la, 'ci_alarm_hi': ha,
                'mean_hd': mh, 'ci_hd_lo': lh, 'ci_hd_hi': hh
            })

        dfagg = pd.DataFrame(agg_rows)
        out_csv = os.path.join(REPORT_DIR, 'experiment_summary_repeats.csv')
        dfagg.to_csv(out_csv, index=False)

        # plot with error bars
        plt.figure(figsize=(6,4))
        plt.errorbar(dfagg['pdr'], dfagg['mean_success'], yerr=[dfagg['mean_success']-dfagg['ci_success_lo'], dfagg['ci_success_hi']-dfagg['mean_success']], fmt='-o', label='Authentication Success (95% CI)')
        plt.errorbar(dfagg['pdr'], dfagg['mean_alarm'], yerr=[dfagg['mean_alarm']-dfagg['ci_alarm_lo'], dfagg['ci_alarm_hi']-dfagg['mean_alarm']], fmt='-s', label='Alarm Rate (95% CI)')
        plt.xlabel('PDR')
        plt.ylabel('Rate')
        plt.title(f'Aggregated Results ({repeats} repeats)')
        plt.grid(True)
        plt.legend()
        out_png = os.path.join(REPORT_DIR, 'experiment_summary_repeats.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        print(f"Saved aggregated CSV to {out_csv} and PNG to {out_png}")
