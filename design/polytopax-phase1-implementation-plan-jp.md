# PolytopAX Phase 1 実装計画書（v0.1.0）

## プロジェクト概要

**PolytopAX v0.1.0**は、JAXエコシステム上で動作する微分可能な凸包計算ライブラリの基盤となるMVP版です。この版では、機械学習パイプラインに統合可能な近似凸包計算アルゴリズムとベーシックな幾何学的操作を提供します。

### 開発目標

- **微分可能性**: JAXの自動微分と完全互換な実装
- **GPU最適化**: XLAコンパイルによる高速計算
- **使いやすさ**: 直感的な関数型・オブジェクト指向API
- **拡張性**: Phase 2以降の機能拡張に対応する設計

## 技術アーキテクチャ

### モジュール構成

```
polytopax/
├── core/
│   ├── __init__.py              # コアモジュールエクスポート
│   ├── hull.py                  # 基本凸包計算（既存、拡張）
│   ├── utils.py                 # 共通ユーティリティ【新規】
│   └── polytope.py              # ConvexHullクラス【新規】
├── algorithms/
│   ├── __init__.py
│   └── approximation.py         # 近似アルゴリズム【新規】
├── operations/
│   ├── __init__.py
│   └── predicates.py            # 幾何述語【新規】
└── __init__.py                  # パッケージエントリポイント
```

### 実装優先度

| 優先度 | モジュール | 内容 | 相互依存性 |
|--------|------------|------|------------|
| **1** | `core/utils.py` | 型定義、バリデーション、方向ベクトル生成 | 独立 |
| **2** | `algorithms/approximation.py` | 微分可能近似凸包 | utils.py |
| **3** | `operations/predicates.py` | 幾何述語（点包含、体積、表面積） | utils.py |
| **4** | `core/polytope.py` | ConvexHullクラス | 上記すべて |
| **5** | `core/hull.py` | 既存関数の拡張とリファクタリング | 上記すべて |

## 詳細技術仕様

### 1. core/utils.py - 基盤ユーティリティ

#### 型定義

```python
from typing import Union, Tuple, Optional, Literal
from jax import Array
import jax.numpy as jnp

# 型エイリアス
PointCloud = Array  # shape: (..., n_points, dimension)
HullVertices = Array  # shape: (n_vertices, dimension)
DirectionVectors = Array  # shape: (n_directions, dimension)
SamplingMethod = Literal["uniform", "icosphere", "adaptive"]
```

#### 核心関数

```python
def validate_point_cloud(points: Array) -> Array:
    """点群の形状・数値妥当性検証
    
    Args:
        points: 入力点群 shape (..., n_points, dim)
        
    Returns:
        検証済み点群
        
    Raises:
        ValueError: 不正な形状または数値
    """

def generate_direction_vectors(
    dimension: int,
    n_directions: int,
    method: SamplingMethod = "uniform",
    random_key: Optional[Array] = None
) -> DirectionVectors:
    """方向ベクトルの生成
    
    Args:
        dimension: 空間次元
        n_directions: 生成する方向数
        method: サンプリング手法
            - "uniform": 球面上一様分布
            - "icosphere": 正20面体細分化（3Dのみ）
            - "adaptive": 局所密度適応サンプリング
        random_key: JAX乱数キー
        
    Returns:
        正規化済み方向ベクトル集合
    """

def robust_orientation_test(
    points: Array,
    tolerance: float = 1e-12
) -> Array:
    """ロバストな幾何学的向き判定
    
    数値誤差に対して安定な orientation test を実装
    Shewchuk (1997) のadaptive precision predicates をベース
    """
```

### 2. algorithms/approximation.py - 微分可能近似凸包

#### 核心アルゴリズム

**方向ベクトルサンプリング手法**:
1. 指定された方向ベクトル集合を生成
2. 各方向で最遠点を計算（微分可能）
3. 重複除去とクリーンアップ

```python
def approximate_convex_hull(
    points: PointCloud,
    n_directions: int = 100,
    method: SamplingMethod = "uniform",
    temperature: float = 0.1,
    random_key: Optional[Array] = None
) -> Tuple[HullVertices, Array]:
    """微分可能な近似凸包計算
    
    Args:
        points: 点群 shape (..., n_points, dim)
        n_directions: サンプリング方向数
        method: サンプリング手法
        temperature: softmax温度（微分可能性調整）
        random_key: 乱数キー
        
    Returns:
        (hull_vertices, hull_indices): 凸包頂点と元インデックス
        
    Algorithm:
        1. 方向ベクトル生成: generate_direction_vectors()
        2. 方向別最遠点探索:
           scores = jnp.dot(points, directions.T)  # shape: (n_points, n_directions)
           weights = jax.nn.softmax(scores / temperature, axis=0)
           soft_points = jnp.sum(weights[..., None] * points, axis=0)
        3. 重複除去: unique_vertices_removal()
    """

def batched_approximate_hull(
    batch_points: Array,
    **kwargs
) -> Tuple[Array, Array]:
    """バッチ処理対応版
    
    Args:
        batch_points: shape (batch_size, n_points, dim)
        
    Returns:
        バッチ化された凸包結果
        
    Implementation:
        return jax.vmap(approximate_convex_hull, in_axes=(0,))(batch_points, **kwargs)
    """
```

#### 数値安定性の確保

```python
def soft_argmax_selection(
    scores: Array,
    temperature: float,
    points: Array
) -> Array:
    """微分可能な最遠点選択
    
    従来のargmaxは微分不可能:
        idx = jnp.argmax(scores)  # 微分不可能
        
    Soft selection による解決:
        weights = softmax(scores / temperature)
        soft_point = sum(weights * points)  # 微分可能
    """
    
def adaptive_temperature_control(
    scores: Array,
    target_sparsity: float = 0.1
) -> float:
    """適応的温度制御
    
    動的に温度を調整し、適切な sparsity を維持
    """
```

### 3. operations/predicates.py - 幾何述語

#### 基本述語

```python
def point_in_convex_hull(
    point: Array,
    hull_vertices: HullVertices,
    tolerance: float = 1e-8
) -> bool:
    """点の凸包内包含判定
    
    Algorithm: 線形計画法による実装
        凸包頂点の凸結合として点が表現可能か判定
        sum(λᵢ * vᵢ) = point, sum(λᵢ) = 1, λᵢ >= 0
    """

def convex_hull_volume(
    vertices: HullVertices,
    method: str = "simplex_decomposition"
) -> float:
    """凸包体積計算（微分可能）
    
    Methods:
        - "simplex_decomposition": 単体分割による積分
        - "monte_carlo": モンテカルロ推定（大規模データ用）
        
    Returns:
        体積値（d次元測度）
    """

def convex_hull_surface_area(
    vertices: HullVertices,
    faces: Optional[Array] = None
) -> float:
    """凸包表面積計算
    
    Args:
        vertices: 凸包頂点
        faces: 面の頂点インデックス（Noneの場合は自動計算）
        
    Algorithm:
        1. 面の抽出（convex_hull_faces()）
        2. 各面の面積計算
        3. 総和
    """
```

#### 距離計算

```python
def distance_to_convex_hull(
    point: Array,
    hull_vertices: HullVertices
) -> float:
    """点から凸包への最短距離
    
    Returns:
        distance: 正値（外部）、0（境界上）、負値（内部）
    """

def hausdorff_distance(
    hull1: HullVertices,
    hull2: HullVertices
) -> float:
    """2つの凸包間のハウスドルフ距離"""
```

### 4. core/polytope.py - ConvexHullクラス

#### クラス設計

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import jax

@dataclass
class ConvexHull:
    """JAX-compatible ConvexHull class
    
    Attributes:
        vertices: 凸包頂点座標
        faces: 面の構成（オプション）
        algorithm_info: 計算メタデータ
        _volume_cache: 体積計算キャッシュ
    """
    vertices: HullVertices
    faces: Optional[Array] = None
    algorithm_info: Dict[str, Any] = None
    _volume_cache: Optional[float] = None
    
    def __post_init__(self):
        """初期化後処理"""
        if self.algorithm_info is None:
            self.algorithm_info = {}
        # JAX tree registration
        jax.tree_util.register_pytree_node(
            ConvexHull,
            self._tree_flatten,
            self._tree_unflatten
        )
    
    # JAX tree compatibility
    def _tree_flatten(self):
        children = (self.vertices, self.faces)
        aux_data = (self.algorithm_info, self._volume_cache)
        return children, aux_data
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        vertices, faces = children
        algorithm_info, volume_cache = aux_data
        return cls(vertices, faces, algorithm_info, volume_cache)
```

#### メソッド実装

```python
class ConvexHull:
    # ... (上記の__init__部分)
    
    def volume(self) -> float:
        """体積計算（キャッシュ対応）"""
        if self._volume_cache is None:
            self._volume_cache = convex_hull_volume(self.vertices)
        return self._volume_cache
    
    def surface_area(self) -> float:
        """表面積計算"""
        return convex_hull_surface_area(self.vertices, self.faces)
    
    def contains(self, point: Array) -> bool:
        """点の包含判定"""
        return point_in_convex_hull(point, self.vertices)
    
    def distance_to(self, point: Array) -> float:
        """点までの距離"""
        return distance_to_convex_hull(point, self.vertices)
    
    def centroid(self) -> Array:
        """重心計算"""
        return jnp.mean(self.vertices, axis=0)
    
    def bounding_box(self) -> Tuple[Array, Array]:
        """バウンディングボックス"""
        min_coords = jnp.min(self.vertices, axis=0)
        max_coords = jnp.max(self.vertices, axis=0)
        return min_coords, max_coords
    
    # 将来のメソッドチェーン用（Phase 2で実装）
    # def scale(self, factor: Union[float, Array]) -> 'ConvexHull': ...
    # def translate(self, vector: Array) -> 'ConvexHull': ...
    # def rotate(self, angle: float, axis: Array = None) -> 'ConvexHull': ...
```

### 5. core/hull.py - 既存コードの拡張

#### 統合API

```python
def convex_hull(
    points: PointCloud,
    algorithm: str = "approximate",
    **kwargs
) -> HullVertices:
    """統一凸包計算インターフェース
    
    Args:
        points: 入力点群
        algorithm: アルゴリズム選択
            - "approximate": 微分可能近似（デフォルト）
            - "quickhull": 正確なQuickhull（Phase 2で実装）
            - "graham_scan": 2D Graham scan（Phase 2で実装）
        **kwargs: アルゴリズム固有パラメータ
        
    Returns:
        凸包頂点配列
    """
    if algorithm == "approximate":
        hull_vertices, _ = approximate_convex_hull(points, **kwargs)
        return hull_vertices
    else:
        raise NotImplementedError(f"Algorithm '{algorithm}' not implemented in v0.1.0")

# 後方互換性のための既存関数の保持
def approximate_convex_hull(
    points: Array,
    n_directions: int = 100,
    method: str = "uniform",
    random_seed: int = 0
) -> Tuple[Array, Array]:
    """既存のシグネチャを維持"""
    # 新しい実装への転送
    from ..algorithms.approximation import approximate_convex_hull as new_impl
    key = jax.random.PRNGKey(random_seed) if random_seed else None
    return new_impl(points, n_directions, method, random_key=key)
```

## API設計とユースケース

### 関数型API（低レベル）

```python
import polytopax as ptx
import jax.numpy as jnp

# 基本的な凸包計算
points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
hull_vertices = ptx.convex_hull(points, algorithm='approximate')

# パラメータ調整
hull_vertices = ptx.approximate_convex_hull(
    points,
    n_directions=200,
    method='icosphere',
    temperature=0.05
)

# バッチ処理
batch_points = jnp.array([...])  # shape: (batch_size, n_points, dim)
batch_hulls = jax.vmap(ptx.convex_hull)(batch_points)

# 基本的な幾何判定
is_inside = ptx.point_in_hull(test_point, hull_vertices)
volume = ptx.hull_volume(hull_vertices)
surface_area = ptx.hull_surface_area(hull_vertices)
```

### オブジェクト指向API（高レベル）

```python
from polytopax import ConvexHull
import jax.numpy as jnp

# ConvexHullオブジェクトの作成
points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
hull = ConvexHull.from_points(points, algorithm='approximate', n_directions=100)

# 幾何学的性質の取得
print(f"体積: {hull.volume():.6f}")
print(f"表面積: {hull.surface_area():.6f}")
print(f"重心: {hull.centroid()}")
print(f"頂点数: {len(hull.vertices)}")

# 幾何判定
test_point = jnp.array([0.5, 0.5])
contains_point = hull.contains(test_point)
distance = hull.distance_to(test_point)

# バウンディングボックス
min_coords, max_coords = hull.bounding_box()
```

### 機械学習統合例

```python
import jax
import jax.numpy as jnp
import polytopax as ptx

def neural_polytope_layer(points, weights):
    """ニューラルネットワーク層としての凸包計算"""
    # アフィン変換
    transformed_points = jnp.dot(points, weights)
    
    # 微分可能な凸包計算
    hull_vertices, _ = ptx.approximate_convex_hull(
        transformed_points,
        n_directions=50,
        temperature=0.1
    )
    
    # 体積を特徴量として返す
    return ptx.hull_volume(hull_vertices)

# 勾配計算
volume_grad = jax.grad(neural_polytope_layer, argnums=1)

# JIT コンパイル
jit_layer = jax.jit(neural_polytope_layer)
```

## JAX統合とパフォーマンス最適化

### JAX変換対応

```python
# JIT コンパイル
@jax.jit
def fast_hull_computation(points):
    return ptx.convex_hull(points, algorithm='approximate')

# 自動微分
@jax.grad
def hull_volume_gradient(points):
    hull_vertices, _ = ptx.approximate_convex_hull(points)
    return ptx.hull_volume(hull_vertices)

# ベクトル化
batched_computation = jax.vmap(fast_hull_computation)

# 並列化（マルチデバイス）
@jax.pmap
def parallel_hull_computation(points_shards):
    return jax.vmap(ptx.convex_hull)(points_shards)
```

### 最適化戦略

1. **XLA最適化**
   - 純粋JAX実装によるコンパイル最適化
   - fused kernel による効率的なメモリアクセス

2. **数値安定性**
   - robust geometric predicates の実装
   - adaptive precision による縮退ケース処理

3. **メモリ効率**
   - in-place操作の活用
   - 不要な中間配列の削減

## テスト戦略

### テストカテゴリ

```python
# 1. 単体テスト
class TestUtils:
    def test_validate_point_cloud(self): ...
    def test_generate_direction_vectors(self): ...
    def test_robust_orientation_test(self): ...

class TestApproximation:
    def test_approximate_convex_hull_2d(self): ...
    def test_approximate_convex_hull_3d(self): ...
    def test_batched_computation(self): ...

class TestPredicates:
    def test_point_in_convex_hull(self): ...
    def test_convex_hull_volume(self): ...
    def test_distance_computation(self): ...

class TestConvexHull:
    def test_from_points_creation(self): ...
    def test_geometric_properties(self): ...
    def test_jax_tree_compatibility(self): ...

# 2. JAX変換テスト
class TestJAXIntegration:
    def test_jit_compilation(self): ...
    def test_gradient_computation(self): ...
    def test_vmap_batching(self): ...
    def test_pmap_parallelization(self): ...

# 3. 数値精度テスト
class TestNumericalAccuracy:
    def test_known_geometries(self):
        """既知の幾何形状での精度検証"""
        # 正方形、立方体、球面など
        
    def test_degenerate_cases(self):
        """縮退ケースの処理"""
        # 共線点、共面点、重複点など

# 4. 性能テスト
class TestPerformance:
    def test_scalability(self): ...
    def test_memory_usage(self): ...
    def test_gpu_acceleration(self): ...
```

### 数値精度の検証

```python
def test_numerical_accuracy():
    """既知解との比較による精度検証"""
    
    # 2D正方形
    square_points = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    hull = ConvexHull.from_points(square_points)
    assert abs(hull.volume() - 1.0) < 1e-6
    assert abs(hull.surface_area() - 4.0) < 1e-6
    
    # 3D立方体
    cube_vertices = jnp.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    hull = ConvexHull.from_points(cube_vertices)
    assert abs(hull.volume() - 1.0) < 1e-6
    assert abs(hull.surface_area() - 6.0) < 1e-6
```

## パフォーマンス目標と評価指標

### 定量的目標

| 指標 | 目標値 | 評価方法 |
|------|--------|----------|
| **計算速度** | SciPy比 2-5倍高速 | 1K-100K点での処理時間比較 |
| **メモリ効率** | 線形スケーリング | 点数増加に対するメモリ使用量 |
| **数値精度** | 相対誤差 < 1e-6 | 解析解との比較 |
| **GPU加速** | CPU比 10-50倍 | 大規模データセットでの比較 |
| **コンパイル時間** | < 5秒 | 初回JIT コンパイル時間 |

### ベンチマーク環境

```python
import time
import psutil
import jax.numpy as jnp
from scipy.spatial import ConvexHull as ScipyHull

def benchmark_comparison():
    """SciPy vs PolytopAX 性能比較"""
    
    sizes = [100, 1000, 10000, 100000]
    dimensions = [2, 3, 4, 5]
    
    results = {}
    
    for n_points in sizes:
        for dim in dimensions:
            # テストデータ生成
            points = jax.random.normal(
                jax.random.PRNGKey(42),
                (n_points, dim)
            )
            
            # SciPy計測
            start_time = time.time()
            scipy_hull = ScipyHull(points)
            scipy_time = time.time() - start_time
            
            # PolytopAX計測
            start_time = time.time()
            ptx_hull = ptx.convex_hull(points, algorithm='approximate')
            ptx_time = time.time() - start_time
            
            results[(n_points, dim)] = {
                'scipy_time': scipy_time,
                'polytopax_time': ptx_time,
                'speedup': scipy_time / ptx_time
            }
    
    return results
```

## 開発スケジュールとマイルストーン

### Week 1-2: 基盤実装
- [ ] `core/utils.py` の実装
- [ ] 基本テストスイートの作成
- [ ] CI/CD パイプラインの設定

### Week 3-4: アルゴリズム実装  
- [ ] `algorithms/approximation.py` の実装
- [ ] 微分可能性の検証
- [ ] バッチ処理対応

### Week 5-6: 幾何述語実装
- [ ] `operations/predicates.py` の実装
- [ ] 数値安定性の確保
- [ ] 距離計算機能

### Week 7-8: ConvexHullクラス
- [ ] `core/polytope.py` の実装
- [ ] JAX tree compatibility
- [ ] オブジェクト指向API

### Week 9-10: 統合とテスト
- [ ] `core/hull.py` の拡張
- [ ] 包括的テストスイート
- [ ] 性能ベンチマーク

### Week 11-12: ドキュメントと品質保証
- [ ] API ドキュメント作成
- [ ] 使用例とチュートリアル
- [ ] リリース準備

## リスク管理と対策

### 技術的リスク

1. **数値安定性の課題**
   - **リスク**: 浮動小数点誤差による予期しない結果
   - **対策**: robust geometric predicates の実装、adaptive precision

2. **JAX制約への対応**
   - **リスク**: JAXの関数型制約による実装難易度
   - **対策**: 段階的実装、豊富なテストケース

3. **パフォーマンス目標未達**
   - **リスク**: SciPyに対する性能優位性が不十分
   - **対策**: プロファイリング主導の最適化、XLA最適化の活用

### プロジェクトリスク

1. **実装スコープの拡大**
   - **リスク**: 機能追加による開発期間延長
   - **対策**: MVP に集中、Phase 2への機能延期

2. **品質保証の不足**
   - **リスク**: テスト不足による潜在バグ
   - **対策**: TDD アプローチ、継続的テスト

## 成功指標

### リリース基準

- [ ] 全単体テスト合格（カバレッジ > 90%）
- [ ] JAX変換（jit, grad, vmap）正常動作
- [ ] ベンチマーク目標達成（SciPy比 2倍以上高速）
- [ ] 数値精度基準満足（相対誤差 < 1e-6）
- [ ] ドキュメント完備

### 技術的KPI

- **コード品質**: mypy, ruff 警告ゼロ
- **テストカバレッジ**: 90%以上
- **性能向上**: SciPy比 2-5倍高速化
- **メモリ効率**: 線形スケーリング維持
- **GPU活用**: 大規模データで 10倍以上高速化

この実装計画に基づき、Phase 1の開発を段階的に進行し、v0.1.0リリースを目指します。