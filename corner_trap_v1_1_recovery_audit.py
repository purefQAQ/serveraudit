#!/usr/bin/env python3
"""BenchPush Corner Trap v1.1: physical recovery challenge after the trap.

Requires corner_trap_v1.py (v1 package) alongside this file. Run from
BenchPush repo root after the v1 scenario has passed.

This is an exploratory *best-effort challenge*, not a proof that recovery is
impossible and not a full B2 policy. Every attempted recovery is executed by
BenchPush physics after independently replaying the same trap sequence.
"""
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from corner_trap_v1 import (
    Runner, WALL_START, START, GOAL, LEFT_WALL, RIGHT_WALL,
    geometric_path_exists, contact_access_certificate,
)


def prepare_identical_trap(seed, budget, outdir):
    """Replay the v1 physical trap and its 12-step physical retreat."""
    r = Runner('trap', outdir, seed=seed, budget=budget)
    ok = r.push_until(math.pi / 2, 13.60, 'push_into_dead_end', max_steps=130)
    r.snapshot('post_interaction')
    if not ok or r.done:
        raise RuntimeError('Failed to construct trap via physical pushing: ' + str(r.end_reason))
    r.phase = 'original_retreat'
    for _ in range(12):
        if not r.step(-0.15, 0.0):
            break
    if r.done:
        raise RuntimeError('Robot terminated during trap setup: ' + str(r.end_reason))
    r.snapshot('after_retreat')
    return r


def go(r, xy, phase, limit=105):
    if not r.done:
        r.goto(xy, phase, max_steps=min(limit, max(1, r.budget - r.total_steps)))


def try_method(r, method, rng):
    """Deliberately simple, reproducible recovery challengers."""
    if method == 'south_reapproach':
        go(r, (11.25, 8.1), 'clear_south', 80)
        go(r, (11.25, 12.8), 'reapproach', 110)
    elif method in ('west_flank', 'east_flank', 'north_flank'):
        side = 'west' if method == 'west_flank' else 'east'
        x = 8.0 if side == 'west' else 14.0
        go(r, (11.25, 7.8), 'clear_entrance', 110)
        go(r, (x, 8.0), 'outside_corridor', 100)
        go(r, (x, 12.6 if method != 'north_flank' else 13.7), 'follow_outer_wall', 120)
        go(r, (11.25, 13.7 if method == 'north_flank' else 12.6), 'try_lateral_contact', 115)
    elif method in ('oblique_left', 'oblique_right'):
        sign = -1 if method == 'oblique_left' else 1
        go(r, (11.25 + sign * 0.45, 11.9), 'staging_oblique', 65)
        if not r.done:
            r.phase = 'oblique_push'
            desired = math.pi / 2 - sign * 0.38
            r.align(desired, max_steps=60)
            for _ in range(38):
                if not r.step(0.15, 0.0):
                    break
    elif method.startswith('random_'):
        # Bursty bounded v/omega commands, deterministic from trial seed.
        # NOT an informed recoverer; useful only as an extra falsification probe.
        for j in range(26):
            if r.done or r.total_steps >= r.budget:
                break
            speed = rng.choice([-0.15, -0.10, 0.0, 0.10, 0.15])
            omega = rng.uniform(-1.0, 1.0)
            r.phase = 'random_burst_%02d' % j
            for _ in range(rng.randint(4, 13)):
                if not r.step(speed, omega):
                    break
    else:
        raise ValueError(method)

    # If the intervention did not already terminate, make a goal attempt.
    # Failing to reach the goal after this finite attempt is NOT a proof of
    # unreachable state under all admissible action sequences.
    if not r.done and r.total_steps < r.budget:
        go(r, (11.25, 13.7), 'goal_attempt', min(150, r.budget - r.total_steps))


def save_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def run_case(seed, budget, outdir, method, trial_index):
    r = prepare_identical_trap(seed, budget, outdir)
    anchor_center = r.box_center().copy()
    anchor_robot = np.array(r.robot(), dtype=float)
    anchor_time = r.total_steps
    anchor_polygon = r.box_polygon().copy()
    rng = random.Random(seed * 100003 + trial_index * 7919)

    try_method(r, method, rng)

    last_polygon = r.box_polygon().copy()
    last_center = r.box_center().copy()
    cspace = geometric_path_exists(last_polygon)
    contact = contact_access_certificate(last_polygon)
    shortest_distance = min(
        [math.hypot(row['robot_x'] - GOAL[0], row['robot_y'] - GOAL[1]) for row in r.traj]
        or [float('inf')]
    )
    name = '%02d_%s' % (trial_index, method)
    csv_path = outdir / (name + '.csv')
    save_csv(csv_path, r.traj)
    record = {
        'method': method,
        'attempt': trial_index,
        'trap_anchor_step': anchor_time,
        'trap_anchor_robot': anchor_robot.tolist(),
        'trap_anchor_box': anchor_center.tolist(),
        'steps_total': r.total_steps,
        'steps_after_trap': r.total_steps - anchor_time,
        'end_reason': r.end_reason or ('budget_exhausted' if r.total_steps >= budget else 'script_completed'),
        'physical_goal_reached': bool(r.info.get('trial_success', False)),
        'wall_collision': bool(r.env.wall_collision),
        'box_delta_from_trap': float(np.linalg.norm(last_center - anchor_center)),
        'box_final_center': last_center.tolist(),
        'goal_min_center_distance': shortest_distance,
        'fixed_box_goal_reachable_after': bool(cspace['reachable']),
        'any_axis_aligned_contact_fits_after': bool(any(contact[k] for k in (
            'north_side_contact_fits', 'west_side_contact_fits', 'east_side_contact_fits'))),
        'max_box_robot_polygon_overlap_area': float(r.max_body_box_overlap_area),
        'trajectory_csv': str(csv_path),
    }
    r.env.close()
    return record, anchor_polygon, last_polygon


def plot_cases(records, boxes, outdir):
    # Draw selected strategies, not all randomized trials.
    candidates = [z for z in records if not z['method'].startswith('random_')]
    random_cases = [z for z in records if z['method'].startswith('random_')]
    if random_cases:
        candidates.append(max(random_cases, key=lambda z: (
            z['physical_goal_reached'], z['box_delta_from_trap'])))
    count = len(candidates)
    ncols = 3
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, rec in zip(axes.flat, candidates):
        ax.set_visible(True)
        name = '%02d_%s' % (rec['attempt'], rec['method'])
        original, final = boxes[name]
        for wall in [
            [(0,0),(15,0)], [(0,0),(0,15)], [(15,0),(15,15)], [(0,15),(15,15)],
            [(LEFT_WALL,WALL_START),(LEFT_WALL,15)],
            [(RIGHT_WALL,WALL_START),(RIGHT_WALL,15)]
        ]:
            seg=np.asarray(wall)
            ax.plot(seg[:,0],seg[:,1], color='black', linewidth=4)
        for poly,color,label in ((original,'darkorange','trap state'),(final,'purple','after attempt')):
            ring=np.vstack([poly, poly[0]])
            ax.plot(ring[:,0],ring[:,1],color=color,linewidth=2,label=label)
        with open(rec['trajectory_csv'], newline='', encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        if rows:
            ax.plot([float(z['robot_x']) for z in rows],
                    [float(z['robot_y']) for z in rows],color='royalblue',linewidth=1.4,label='physical trajectory')
        ax.scatter([GOAL[0]],[GOAL[1]],c='red',marker='*',s=100)
        ax.add_patch(plt.Circle(GOAL,.86,edgecolor='red',fill=False,linestyle=':'))
        ax.set_title('%s | reached=%s' % (rec['method'],rec['physical_goal_reached']),fontsize=10)
        ax.set_xlim(0,15);ax.set_ylim(0,15);ax.set_aspect('equal')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7,loc='lower left')
    fig.suptitle('Corner Trap v1.1: finite physical recovery challengers')
    fig.tight_layout()
    fig.savefig(outdir/'recovery_comparison.png',dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--budget',type=int,default=400)
    p.add_argument('--random-trials',type=int,default=4)
    p.add_argument('--outdir',default='logs/corner_trap_v1_1')
    args=p.parse_args()
    if args.budget < 120:
        p.error('--budget must allow the original trap plus meaningful recovery attempts')
    outdir=Path(args.outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    methods=['south_reapproach','west_flank','east_flank','north_flank',
             'oblique_left','oblique_right']
    methods += ['random_%02d' % i for i in range(args.random_trials)]
    records=[]; boxes={}
    base_robot=None;base_box=None;base_step=None
    for i, method in enumerate(methods):
        record, anchor_box, final_box=run_case(args.seed,args.budget,outdir,method,i)
        name='%02d_%s'%(i,method)
        boxes[name]=(anchor_box,final_box)
        records.append(record)
        if base_robot is None:
            base_robot=np.asarray(record['trap_anchor_robot'])
            base_box=np.asarray(record['trap_anchor_box'])
            base_step=record['trap_anchor_step']
        else:
            # Fairly replay the same trap state for each challenger.
            if not (np.allclose(base_robot,record['trap_anchor_robot'],atol=1e-5)
                    and np.allclose(base_box,record['trap_anchor_box'],atol=1e-5)
                    and base_step==record['trap_anchor_step']):
                raise AssertionError('Trap replay diverged: %s' % method)
        print('[TRIAL]',method,
              'goal=',record['physical_goal_reached'],
              'wall=',record['wall_collision'],
              'box_delta=',round(record['box_delta_from_trap'],3),
              'cspace=',record['fixed_box_goal_reachable_after'],
              'steps=',record['steps_total'],flush=True)
    results={
        'scenario':'corner_trap_v1_1_recovery_challenge',
        'seed':args.seed,
        'budget_per_trial_including_trap':args.budget,
        'same_trap_state_replayed':True,
        'experiment_type':'scripted_physical_recovery_challengers_NOT_full_B2',
        'physical_recovery_impossibility_proved':False,
        'uncertainty_or_partial_observability_present':False,
        'any_physical_goal_witness_found':any(z['physical_goal_reached'] for z in records),
        'any_fixed_box_cspace_restored':any(z['fixed_box_goal_reachable_after'] for z in records),
        'attempts':records,
    }
    with open(outdir/'summary.json','w',encoding='utf-8') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    plot_cases(records,boxes,outdir)
    print('[SUMMARY] physical success witnesses =',sum(z['physical_goal_reached'] for z in records))
    print('[SUMMARY] cspace restored witnesses =',sum(z['fixed_box_goal_reachable_after'] for z in records))
    print('[WARNING] Zero witnesses does NOT establish irreversibility or outperforming B2.')
    print('[FILES]',outdir.resolve())


if __name__=='__main__':
    main()
