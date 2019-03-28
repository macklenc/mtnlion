"""Generate statistics for baseline"""

import numpy as np
import pickle
import click
import multiprocessing

time = np.arange(0.1, 50, 1)
sim_dt = 0.1


def gen_phase1(return_dict):
    from buildup.fenics_.phase1 import (phis, phie, cs, ce, j)

    return_dict['phase1_'] = {
        'phis': phis.main(time=time, get_test_stats=True),
        'phie': phie.main(time=time, get_test_stats=True),
        'cs': cs.main(time=time, get_test_stats=True),
        'ce': ce.main(time=time, get_test_stats=True),
        'j': j.main(time=time, get_test_stats=True),
    }


def gen_phase1t(return_dict):
    from buildup.fenics_.phase1t import (cs, ce)

    return_dict['phase1t_'] = {
        'cs': cs.main(start_time=time[0], dt=sim_dt, stop_time=time[-1], plot_time=time, get_test_stats=True),
        'ce': ce.main(start_time=time[0], dt=sim_dt, stop_time=time[-1], plot_time=time, get_test_stats=True),
    }


def gen_phase2(return_dict):
    from buildup.fenics_.phase2 import (phis_newton, phie, cs, ce)

    return_dict['phase2_'] = {
        'phis': phis_newton.main(time=time, dt=sim_dt, get_test_stats=True),
        'phie': phie.main(time=time, dt=sim_dt, get_test_stats=True),
        'cs': cs.main(time=time, dt=sim_dt, get_test_stats=True),
        'ce': ce.main(time=time, dt=sim_dt, get_test_stats=True),
    }


def gen_phase2t(return_dict):
    from buildup.fenics_.phase2t import (cs, ce)

    return_dict['phase2t_'] = {
        'cs': cs.main(start_time=time[0], dt=sim_dt, stop_time=time[-1], plot_time=time, get_test_stats=True),
        'ce': ce.main(start_time=time[0], dt=sim_dt, stop_time=time[-1], plot_time=time, get_test_stats=True),
    }

@click.command()
@click.argument('path')
def main(path):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = [
        multiprocessing.Process(target=gen_phase1, args=(return_dict,)),
        multiprocessing.Process(target=gen_phase1t, args=(return_dict,)),
        multiprocessing.Process(target=gen_phase2, args=(return_dict,)),
        multiprocessing.Process(target=gen_phase2t, args=(return_dict,))
    ]

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    d = dict(return_dict)
    data = dict()

    for k, v in d.items():
        for k1, v1 in v.items():
            data[k+k1] = v

    data.update({
        'time': time,
        'sim_dt': sim_dt,
    })

    with open(path, "wb") as file:
        pickle.dump(d, file)

    pass


if __name__ == '__main__':
    main()
