import numpy as np

def test_env_util():
    import envs.env_util as util
    assert(util.wrap_around(3*np.pi/2) == -np.pi/2)
    print("PASSED wrap_around()")

    assert(util.xyg2polarb_dot(np.array([2,3]), np.array([1,0]), np.array([0,0]), 0.0, 0.0, 0.0)
        == util.xydot_to_vw(2.0, 3.0, 1.0, 0.0))
    print("PASSED xyg2polarb_dot(), xydot_to_vw()")

if __name__ == "__main__":
    print("TEST ENV_UTIL.PY...")
    test_env_util()
