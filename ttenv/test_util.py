import numpy as np
import util

def test_env_util():
    qa = {(1,0):(1,0), (0,1):(1,np.pi/2), (-1,0):(1, np.pi), (0,-1):(1, -np.pi/2)}
    for (k,v) in qa.items():
        r, alpha = util.cartesian2polar(np.array(k))
        assert(r == v[0])
        assert(alpha == v[1])
    print("PASSED cartesian2polar()")

    x_b, y_b = util.transform_2d([1,1], np.pi/2)
    assert(np.abs(x_b - 1.0) < 1e-5)
    assert(np.abs(y_b + 1.0) < 1e-5)
    print("PASSED transform_2d()")

    r_t_b, theta_t_b = util.relative_distance_polar(np.array([0, 2]), np.array([1, 1]), np.pi/2)
    assert(np.abs(r_t_b - np.sqrt(2)) < 1e-5)
    assert(np.abs(theta_t_b - np.pi/4) < 1e-5)
    print("PASSED relative_distance_polar()")

    assert(util.wrap_around(3*np.pi/2) == -np.pi/2)
    print("PASSED wrap_around()")

if __name__ == "__main__":
    print("TEST ENV_UTIL.PY...")
    test_env_util()
