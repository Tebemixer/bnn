# Отчёт: как модифицируется градиент в бинарно-квантизированных частях BHViT

## 1) Где именно в архитектуре применяется бинаризация

В `BHViTSelfAttention` бинаризация включается при `config.input_bits == 1`:

- `query/key/value` активности проходят через `BinaryQuantizer.apply(...)`.
- вероятности внимания (`attention_probs`) дополнительно квантуются через:
  - `GSB_Attention.apply(...)` (если `config.gsb=True`), или
  - модуль `BinaryActivation_Attention(...)` (если `config.gsb=False`).

Это задаёт основной «бинарный» путь в attention-блоке.

## 2) Базовый бинарный квантизатор (`BinaryQuantizer`) и его surrogate-gradient

### Forward
`BinaryQuantizer.forward`: `out = sign(input)`.

### Backward
Вместо нулевого градиента от `sign` реализован кусочно-линейный surrogate-gradient:

- для `x in [-1, 0]`: множитель к градиенту `2 + 2x`;
- для `x in (0, 1]`: множитель `2 - 2x`;
- вне диапазона `[-1, 1]`: градиент = 0.

Итог:
`dL/dx = g(x) * dL/dq`, где `g(x)` — «треугольное окно» с максимумом около нуля.

Практический смысл: градиент концентрируется около порога бинаризации и подавляется на насыщенных значениях.

## 3) Как модифицируется градиент для бинарных весов в слоях Linear/Conv/Embedding

Для `weight_bits == 1` в `QuantizeLinear`, `QuantizeConv2d`, `QuantizeConv2d2`, `QuantizeEmbedding` используется STE-подобная конструкция:

1. Строится бинарный вес без градиента:
   `binary_weights_no_grad = scaling_factor * sign(real_weights)`.
2. Строится «клипнутая» непрерывная версия:
   `cliped_weights = clamp(real_weights, -tau, tau)` (`tau=1` или адаптивный `Q_tau` при `recu`).
3. Финальный вес:
   `weight = binary_no_grad.detach() - cliped.detach() + cliped`.

Следствие для backward:

- по прямому проходу используется бинарный вес;
- градиент течёт **только через `cliped_weights`** (т.е. как через `clamp`):
  - внутри диапазона клипа множитель ~1;
  - вне диапазона клипа множитель 0;
- `scaling_factor` и бинарная ветка с `sign` отделены через `detach`, поэтому напрямую в этот путь градиент не идёт.

Итого это STE с «маской насыщения» через clamp.

## 4) Модификация градиента в attention probability quantization

### Вариант A: `GSB_Attention`
`GSB_Attention` делит вероятности на 3 зоны масками (`T0/T1/T2`) и учит параметры `alpha, alpha2, alpha3`.

В backward:

- вычисляются отдельные градиенты для `alpha/alpha2/alpha3` через суммы по маскам;
- для входа используется масштабированный градиент:
  `grad_input = alpha * (T0 + alpha2*T1 + alpha3*T2) * grad_output`.

То есть градиент по `attention_probs` не просто пропускается, а **перевзвешивается зональными коэффициентами**.

### Вариант B: `BinaryActivation_Attention`
Используется LSQ-подобная схема:

- `grad_scale`: задаёт уменьшенный/нормированный градиент для `alpha` и `zero_point` через
  `y.detach() - y_grad.detach() + y_grad`;
- `round_pass`: в forward делает округление, а в backward подставляет градиент как у identity (STE);
- коэффициент масштаба градиента параметров: `g = num_head / sqrt(numel * Qp)`.

Итого для attention-активаций: дискретизация в forward, а в backward — управляемый surrogate-gradient + нормировка градиента параметров квантизатора.

## 5) Краткая карта «часть архитектуры -> как модифицируется градиент»

- **Q/K/V бинаризация (`BinaryQuantizer`)**: треугольный surrogate-gradient в окне `[-1,1]`, ноль вне окна.
- **Бинарные веса линейных/свёрточных/эмбеддинг-слоёв**: STE через `detach`-трюк; градиент фактически как у `clamp` по `real_weights`.
- **Квантование attention probabilities (`GSB_Attention`)**: зональная перевзвеска `grad_input` + обучаемые градиенты `alpha*`.
- **Квантование attention probabilities (`BinaryActivation_Attention`)**: STE через `round_pass`, а градиенты scale/zero-point нормируются `grad_scale`.

## 6) Вывод

В этом коде бинаризация не использует «чистый» градиент `sign` (он почти везде нулевой), а заменяет его на комбинацию:

1. surrogate-gradient оконного типа (`BinaryQuantizer.backward`),
2. STE через `detach` для бинарных весов,
3. mask-aware / scale-aware градиентные правила в квантизаторе attention вероятностей.

Поэтому обучение устойчиво: дискретный forward сохраняется, но backward остаётся информативным в «рабочем» диапазоне значений.
