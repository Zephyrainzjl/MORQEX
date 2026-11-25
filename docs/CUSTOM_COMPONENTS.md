# 自定义微结构、CBD材料、算子和退化接口

接口是可替换的物理契约；不满足当前后端能力的组合会拒绝。新公式必须重新验证守恒、能量、离散误差和数值稳定性。完整方法范围及失败记录见[退化与CT说明](EXTENSIBLE_DEGRADATION.md)。所有示例在受限launcher内执行，输出只写外部workspace。

## CBD不是单一材料常数

几何的`phase_kinds`保留粗语义`pore/active/cbd/inert`；任意多个标签可以属于同一类，各标签用`PhaseCatalog`提供不同物性。例如活性晶粒、CBD、集流体或惰性夹杂可以具有不同固定弹性张量，CBD还可同时含微孔电解液。

```python
from morqex.materials import Phase, PhaseCatalog

catalog = PhaseCatalog({
    0: Phase('electrolyte', 'pore', ionic_efficiency=1., electrolyte_fraction=1.),
    1: Phase.isotropic(
        'active', 'active', young=15e9, poisson=.3,
        chemical_strain_per_concentration=(1e-7, 1e-7, 1e-7, 0., 0., 0.),
        thermal_strain_per_K=(1e-5, 1e-5, 1e-5, 0., 0., 0.),
        electronic_efficiency=.4, diffusion_factor_XYZ=(1., 2., 1.),
        source='synthetic example, not calibrated',
    ),
    2: Phase.isotropic(
        'microporous CBD', 'cbd', young=1e9, poisson=.3,
        ionic_efficiency=.05, electrolyte_fraction=.47,
        electronic_efficiency=1., source='synthetic example, not calibrated',
    ),
})
```

弹性张量可直接传6×6正定Mandel矩阵；化学/热膨胀可传六分量。电子、离子系数相对于DFN参考物性归一化，不能把S/m直接填到无量纲效率中。CBD微孔率与有效离子系数不是同一个量。对角扩散倍率乘共同的D(c,T)，非对角各向异性需要MPFA/FEM离散插件；不同宿主化学需要多物种粒子插件。黏弹/塑性CBD需要历史变量与对应能量/耗散，不能沿用固定线弹性H。

## 组合几何、基与算子

```python
from morqex.workflow import ElectrodeRecipe, CoupledRecipe
from morqex.ct import SegmentedCT
from morqex.bases import PortBasis, VariationalOperator
from morqex.degradation import DegradationOptions

# actual_path / actual_sha256 / crop 必须来自实际下载清单，不能用占位hash运行。
geometry = SegmentedCT(
    path=actual_path, sha256=actual_sha256,
    spacing_m=(398e-9,)*3, phase_kinds={0:'pore', 1:'active', 2:'cbd'},
    axes_to_zyx=(1,2,0), crop=crop,
)
positive = ElectrodeRecipe(
    geometry=geometry, basis=PortBasis(), operator=VariationalOperator(), regions=8,
    topology=catalog, mechanical_parameters={'materials':catalog},
    cohesive_parameters={'normal_stiffness':1e15, 'tangent_stiffness':5e14},
)
# negative 需用相应负极几何/材料另行构建，不能把NMC当作石墨。
recipe = CoupledRecipe(negative=negative, positive=positive,
                       degradation=DegradationOptions())
paths = recipe.prepare(workspace, parameters, 'my_case')
model, stack, kernels = recipe.model(paths, parameters)
```

上述为组合接口示例，不是该参数/几何已通过资格的声明。`electrode_factory`、`cohesive_factory`、`basis`、`operator`以及各自参数都可替换。`CoupledRecipe.submodel_factories`可替换`states/mechanics/kinetics/particle/porosity/transport/thermal`角色；更改状态或物理假设后，必须提供所有消费方需要的变量、导数与单位。当前默认装配是区域浓度形式，不能直接把负值POD系数当作物理浓度送入BV。

已有`concentration.ConcentrationBasis`提供谱、POD、联合度量等线性空间；`concentration_modes.tail.TailBasis`是新的应力观测量区域候选。可注册为自定义基，但没有最优性或普遍P99认证结论。自定义`OperatorBuilder.build(electrode, partition)`返回的能量/场算子需通过对称、半正定与质量检查；新增路径相关能量需同时替换相应力学与热子模型。

## 自定义SEI定律并保留库存契约

```python
from dataclasses import replace
import pybamm as pb
from morqex.degradation import DiffusionLimitedSEI, PlatingStrippingDeadLi
from morqex.extensions import ComponentSpec
from morqex.degradation import standard_registry, DegradationOptions

class ArrheniusSEI(DiffusionLimitedSEI):
    def evaluate(self, context):
        result = super().evaluate(context)
        factor = pb.exp(20000/context.gas_constant *
                        (1/298.15 - 1/context.temperature))
        return replace(result, current=result.current*factor,
                       rates={name:rate*factor for name,rate in result.rates.items()})

registry = standard_registry()
registry.register('sei', ComponentSpec('my-SEI', ArrheniusSEI,
    assumptions=('synthetic activation energy',),
    units={'current':'A/m2', 'reservoir rate':'mol Li/m3/s'}))
config = DegradationOptions(negative=(registry.create('sei','my-SEI'),
                                      PlatingStrippingDeadLi()))
```

同时缩放电流与库存变化，保留副反应化学计量；只改电流或只改库存会破坏守恒。实际计算应保存`registry.manifest()`及全部配置。自定义工厂需可追踪源码；模型的单位、状态范围、自由能与耗散适用性仍需独立验证，不能只凭注册成功认定正确。

## 逐步演算、检查点与拒绝不兼容证书

```python
from morqex.runtime.aging import AgingRuntime
run = AgingRuntime(paths, parameters, config, nx=3)
state = run.advance(20., current_A=.5)
checkpoint = run.checkpoint(workspace.path('checkpoints','my_case'))
fields = run.recover('positive', x_index=1)
restart = AgingRuntime(paths, parameters, config, nx=3).restore(checkpoint)
```

检查点保存所有微分/代数数值、参数/工厂/模型/布局指纹与SHA256，不使用pickle。库存违反契约的自定义定律会被拒绝并回退本步状态。旧固定H反应上界和浓度单场富集不适用于变化cohesive/副反应库存，`AgingRuntime`会明确拒绝这两个旧入口。退化状态的跨基富集尚需独立实现，不把接口留空当作已经解决。
