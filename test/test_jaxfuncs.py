from sbmltoodejax import jaxfuncs
import jax.numpy as jnp
from jax.lax import complex

def test_sec():
    assert jnp.isclose(jaxfuncs.sec(0), 1.0)
    assert jnp.isclose(jaxfuncs.sec(jnp.pi), -1.0)
    assert jnp.isclose(jaxfuncs.sec(2*jnp.pi), 1.0)

def test_csc():
    assert jnp.isclose(jaxfuncs.csc(-jnp.pi/2), -1.0)
    assert jnp.isclose(jaxfuncs.csc(jnp.pi/2), 1.0)
    assert jnp.isclose(jaxfuncs.csc(3*jnp.pi/2), -1.0)

def test_cot():
    assert jnp.isclose(jaxfuncs.cot(jnp.pi/6), jnp.sqrt(3))
    assert jnp.isclose(jaxfuncs.cot(jnp.pi/4), 1.0)
    assert jnp.isclose(jaxfuncs.cot(jnp.pi/3), 1/jnp.sqrt(3))
    assert jnp.isclose(jaxfuncs.cot(jnp.pi/2), 0., atol=1e-6, rtol=0.)

def test_sech():
   assert jnp.isclose(jaxfuncs.sech(0), 1.0)
   assert jnp.isclose(jaxfuncs.sech(complex(0., jnp.pi/6.)), complex(2./jnp.sqrt(3.), 0.))
   assert jnp.isclose(jaxfuncs.sech(complex(0., jnp.pi/4.)), complex(jnp.sqrt(2.), 0.))
   assert jnp.isclose(jaxfuncs.sech(complex(0., jnp.pi/3.)), complex(2., 0.))

def test_csch():
    assert jnp.isclose(jaxfuncs.csch(complex(0., jnp.pi/6.)), complex(0., -2.))
    assert jnp.isclose(jaxfuncs.csch(complex(0., jnp.pi/4.)), complex(0., -jnp.sqrt(2.)))
    assert jnp.isclose(jaxfuncs.csch(complex(0., jnp.pi/3.)), complex(0., -2./jnp.sqrt(3.)))
    assert jnp.isclose(jaxfuncs.csch(complex(0., jnp.pi/2.)), complex(0., -1.))

def test_coth():
    assert jnp.isclose(jaxfuncs.coth(complex(0., jnp.pi/2)), 0., atol=1e-6, rtol=0.)

def test_sigmoid():
    assert jnp.isclose(jaxfuncs.sigmoid(0.), 0.5)

def test_piecewise():
    x = 2
    assert jaxfuncs.piecewise(x + 2, x < 3, x + 4, x > 3) == 4