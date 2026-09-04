from dataclasses import dataclass

REQUIRED_KEYS = ("M", "rho_pure", "p_sat", "Dg")


@dataclass(frozen=True)
class Component:
    """Pure-substance properties of one component."""

    name: str
    M: float          # molar mass, g/mol
    rho_pure: float   # pure liquid density, kg/m^3
    p_sat: float      # saturation vapour pressure at T, Pa
    Dg: float         # gas-phase diffusivity, m^2/s

    def alpha(self, D_ref):
        return D_ref / self.Dg

    def gamma(self, Ml0, rhol_0, R, T):
        return self.p_sat * Ml0 / (rhol_0 * R * T)


class ComponentLibrary:
    """Component properties and binary diffusivities, from caller-supplied dicts."""

    def __init__(self, components, diffusivities=None):
        if not components:
            raise ValueError(
                "ComponentLibrary needs at least one component. Pass a dict of "
                f"name -> {{{', '.join(REQUIRED_KEYS)}}}.")
        for name, data in components.items():
            missing = [k for k in REQUIRED_KEYS if k not in data]
            if missing:
                raise KeyError(
                    f"component {name!r} is missing {missing}. "
                    f"Every entry needs {list(REQUIRED_KEYS)}.")
        self.components = {n: dict(d) for n, d in components.items()}
        self.diffusivities = dict(diffusivities or {})

    def names(self):
        return sorted(self.components)

    def pairs(self):
        return sorted(self.diffusivities)

    def component(self, name, **overrides):
        if name not in self.components:
            raise KeyError(
                f"unknown component {name!r}. Known: {', '.join(self.names())}. "
                f"Add it to the `components` dict you passed to "
                f"ComponentLibrary.")
        data = {k: v for k, v in self.components[name].items()
                if k in REQUIRED_KEYS}
        data.update(overrides)
        return Component(name=name, **data)

    def binary_diffusivities(self, name_A, name_B):
        # A-in-B first, then B-in-A. 
        # the solver marches on, so the pair ORDER rescales the problem.
        try:
            return (self.diffusivities[(name_A, name_B)],
                    self.diffusivities[(name_B, name_A)])
        except KeyError:
            raise KeyError(
                f"no binary diffusivity for ({name_A!r}, {name_B!r}). "
                )
            
