import dbcbs_py
from pathlib import Path
import tempfile
import diffmp


def main():

    time_limit_db_astar = 1000
    time_limit_db_cbs = 1000
    # dynamics = "unicycle1_v0"
    dynamics = "unicycle2_v0"
    config = {
        "delta_0": 0.5,
        "delta_rate": 0.9,
        "num_primitives_0": 100,
        "num_primitives_rate": 1.5,
        "alpha": 0.5,
        "filter_duplicates": True,
        "heuristic1": "reverse-search",
        "heuristic1_delta": 1.0,
        "mp_path": f"../new_format_motions/{dynamics}/{dynamics}.msgpack",
        # "mp_path": f"data/output/unicycle2_v0/rand_0.yaml",
    }
    instance = diffmp.problems.Instance.from_yaml(Path("../example/bugtrap_2.yaml"))
    tmp1 = tempfile.NamedTemporaryFile()
    tmp2 = tempfile.NamedTemporaryFile()
    result = dbcbs_py.db_cbs(
        instance.to_dict(),
        tmp1.name,
        tmp2.name,
        config,
        time_limit_db_astar,
        time_limit_db_cbs,
    )
    tmp1.close()
    tmp2.close()
    print([len(r.optimized.trajectories[0].actions) for r in result])
    print([len(r.discrete.trajectories[0].actions) for r in result])
    breakpoint()


if __name__ == "__main__":
    main()
