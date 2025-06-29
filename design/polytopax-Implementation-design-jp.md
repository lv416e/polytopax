# PolytopAX 実装設計書（日本語版）

## プロジェクト概要

**PolytopAX**は、JAXエコシステム上で動作する高性能な凸包計算・ポリトープ操作ライブラリです。GPU/TPUアクセラレーション、自動微分、バッチ処理などJAXの強力な機能を活用し、既存のCPUベースライブラリ（SciPy、Qhull等）の限界を超えた高速かつ柔軟な幾何学計算を実現します。

### 目標とビジョン

- **高速性**: JAX/XLAによるGPU/TPU最適化で大規模データ処理を実現
- **微分可能性**: 機械学習パイプラインに統合可能な自動微分対応
- **使いやすさ**: 直感的なAPIと豊富なドキュメント
- **拡張性**: 将来のリーマン多様体最適化ライブラリ（GeomAX）への統合も見据えた設計

## アーキテクチャ設計

### モジュール構成

```
polytopax/
├── __init__.py                      # パッケージエントリポイント
├── core/                           # コア機能
│   ├── __init__.py
│   ├── hull.py                     # 凸包計算の基本関数
│   ├── polytope.py                 # Polytopeクラス
│   └── utils.py                    # 共通ユーティリティ
├── algorithms/                     # アルゴリズム実装
│   ├── __init__.py
│   ├── quickhull.py               # Quickhullアルゴリズム
│   ├── graham_scan.py             # Graham scanアルゴリズム
│   ├── approximation.py           # 近似アルゴリズム
│   └── incremental.py             # 増分的アルゴリズム
├── operations/                     # ポリトープ操作
│   ├── __init__.py
│   ├── predicates.py              # 包含判定・幾何述語
│   ├── metrics.py                 # 体積・表面積計算
│   ├── transformations.py         # アフィン変換
│   └── intersection.py            # 交差・合成操作
├── visualization/                  # 可視化ツール
│   ├── __init__.py
│   ├── plotters.py                # 基本プロット機能
│   └── interactive.py             # インタラクティブ可視化
├── benchmarks/                     # 性能評価
│   ├── __init__.py
│   ├── comparison.py              # 他ライブラリとの比較
│   └── profiling.py               # パフォーマンス分析
└── examples/                       # 使用例
    ├── __init__.py
    ├── basic_usage.py
    ├── machine_learning.py
    └── robotics.py
```

### 設計原則

1. **ハイブリッドAPI**: 関数型とオブジェクト指向の両方をサポート
2. **JAXファースト**: すべてのコア計算はJAXで実装
3. **型安全性**: Type hintsとruntime validation
4. **パフォーマンス重視**: XLAコンパイル最適化を前提とした設計
5. **テスト駆動**: 包括的なテストスイート

## API設計

### 関数型API（低レベル）

```python
import polytopax as ptx
import jax.numpy as jnp

# 基本的な凸包計算
points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
hull_vertices = ptx.convex_hull(points, algorithm='quickhull')

# 近似凸包（微分可能）
approx_hull = ptx.approximate_convex_hull(points, n_directions=100)

# バッチ処理
batch_points = jnp.array([...])  # shape: (batch_size, n_points, dim)
batch_hulls = jax.vmap(ptx.convex_hull)(batch_points)

# 点の包含判定
is_inside = ptx.point_in_hull(test_point, hull_vertices)

# 体積計算
volume = ptx.hull_volume(hull_vertices)
```

### オブジェクト指向API（高レベル）

```python
from polytopax import ConvexHull
import jax.numpy as jnp

# ConvexHullオブジェクトの作成
points = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
hull = ConvexHull(points, algorithm='quickhull')

# メソッドチェーンによる操作
transformed_hull = hull.scale(2.0).translate([1, 1]).rotate(jnp.pi/4)

# 幾何学的性質の取得
print(f"Volume: {hull.volume()}")
print(f"Surface area: {hull.surface_area()}")
print(f"Vertices: {hull.vertices()}")

# 他のhullとの操作
other_hull = ConvexHull(other_points)
intersection = hull.intersection(other_hull)
minkowski_sum = hull.minkowski_sum(other_hull)

# 点の包含判定
contains_point = hull.contains(test_point)
```

## コア機能の実装計画

### Phase 1: 基本機能（v0.1.0）

#### 1.1 近似凸包計算
```python
def approximate_convex_hull(
    points: Array,
    n_directions: int = 100,
    method: str = 'uniform',
    random_seed: int = 0
) -> Tuple[Array, Array]:
    """微分可能な近似凸包計算

    Args:
        points: 点群 (shape: [..., n_points, dim])
        n_directions: サンプリング方向数
        method: サンプリング手法 ('uniform', 'adaptive', 'icosphere')
        random_seed: 乱数シード

    Returns:
        hull_points: 凸包頂点
        hull_indices: 元配列でのインデックス
    """
```

#### 1.2 基本的な幾何述語
```python
def point_in_hull(point: Array, hull_vertices: Array, tolerance: float = 1e-6) -> bool:
    """点の凸包内包含判定"""

def hull_volume(vertices: Array) -> float:
    """凸包の体積計算（微分可能）"""

def hull_surface_area(vertices: Array) -> float:
    """凸包の表面積計算"""
```

#### 1.3 ConvexHullクラス
```python
@dataclass
class ConvexHull:
    vertices: Array
    facets: Optional[Array] = None
    algorithm_info: Dict[str, Any] = field(default_factory=dict)

    def contains(self, point: Array) -> bool:
        """点の包含判定"""

    def volume(self) -> float:
        """体積計算"""

    def surface_area(self) -> float:
        """表面積計算"""
```

### Phase 2: 拡張機能（v0.2.0）

#### 2.1 正確な凸包計算
- Quickhull アルゴリズム（2D/3D）
- Graham scan アルゴリズム（2D）
- 増分的凸包計算

#### 2.2 変換操作
```python
def transform_hull(hull: ConvexHull, matrix: Array, translation: Array = None) -> ConvexHull:
    """アフィン変換の適用"""

class ConvexHull:
    def scale(self, factor: Union[float, Array]) -> 'ConvexHull':
        """スケーリング変換"""

    def translate(self, vector: Array) -> 'ConvexHull':
        """平行移動"""

    def rotate(self, angle: float, axis: Array = None) -> 'ConvexHull':
        """回転変換"""
```

#### 2.3 高度なサンプリング戦略
- 適応的サンプリング
- icosphere基盤サンプリング
- ユーザ定義方向ベクトル

### Phase 3: 高度な機能（v0.3.0+）

#### 3.1 複合操作
```python
def minkowski_sum(hull1: ConvexHull, hull2: ConvexHull) -> ConvexHull:
    """ミンコフスキー和の計算"""

def hull_intersection(hull1: ConvexHull, hull2: ConvexHull) -> ConvexHull:
    """凸包の交差計算"""
```

#### 3.2 高次元幾何学
- ボロノイ図生成
- ドロネー三角形分割
- 非凸形状の凸分解

## JAX統合とパフォーマンス最適化

### JAX変換対応

```python
# JIT コンパイル
@jax.jit
def batched_hull_volumes(batch_points):
    return jax.vmap(lambda pts: hull_volume(convex_hull(pts)))(batch_points)

# 勾配計算
@jax.grad
def hull_volume_gradient(points):
    hull_vertices = approximate_convex_hull(points)[0]
    return hull_volume(hull_vertices)

# 並列化
@jax.pmap
def parallel_hull_computation(points_shards):
    return jax.vmap(convex_hull)(points_shards)
```

### 最適化戦略

1. **XLA最適化**: 純粋JAX実装によるコンパイル最適化
2. **メモリ効率**: in-place操作とメモリプールの活用
3. **数値安定性**: robust geometric predicatesの実装
4. **スケーラビリティ**: 大規模データに対応したアルゴリズム選択

## テスト戦略

### テストカテゴリ

1. **単体テスト**: 各関数・メソッドの動作確認
2. **統合テスト**: モジュール間の連携テスト
3. **回帰テスト**: 既知の問題の再発防止
4. **性能テスト**: ベンチマークとプロファイリング
5. **数値テスト**: 数値安定性と精度の検証

### テスト環境

```python
# pytest + JAXテストユーティリティ
import pytest
import jax.test_util as jtu
from polytopax.testing import assert_hulls_close

class TestConvexHull:
    @pytest.mark.parametrize("algorithm", ["quickhull", "approximate"])
    def test_square_hull(self, algorithm):
        points = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        hull = ConvexHull(points, algorithm=algorithm)
        assert hull.volume() == pytest.approx(1.0)

    def test_gradient_computation(self):
        def objective(points):
            return ConvexHull(points).volume()

        points = jnp.array([[0, 0], [1, 0], [0, 1]])
        grad_fn = jax.grad(objective)
        gradients = grad_fn(points)
        assert gradients.shape == points.shape
```

## ドキュメント戦略

### ドキュメント構成

1. **README**: プロジェクト概要と基本的な使用方法
2. **Installation Guide**: インストールと環境設定
3. **API Reference**: 詳細なAPI仕様書
4. **User Guide**: 段階的なチュートリアル
5. **Examples**: 実用的な使用例集
6. **Developer Guide**: 開発者向けドキュメント

### サンプルコード

```python
# 機械学習での使用例
import polytopax as ptx
import jax.numpy as jnp
from jax import grad, jit

def neural_polytope_layer(points, weights):
    """ニューラルネットワーク層としての凸包計算"""
    transformed_points = jnp.dot(points, weights)
    hull = ptx.approximate_convex_hull(transformed_points)
    return ptx.hull_volume(hull[0])

# 勾配を自動計算
volume_grad = jit(grad(neural_polytope_layer, argnums=1))
```

## 開発ロードマップ

### マイルストーン

**v0.1.0 (MVP) - 3ヶ月**
- [ ] 基本的な近似凸包計算
- [ ] ConvexHullクラス
- [ ] 基本的な幾何述語
- [ ] 包括的なテストスイート
- [ ] 基本ドキュメント

**v0.2.0 (拡張) - 6ヶ月**
- [ ] 正確な凸包アルゴリズム
- [ ] 変換操作
- [ ] 可視化ツール
- [ ] 性能ベンチマーク
- [ ] 詳細なAPI仕様書

**v0.3.0 (高度機能) - 12ヶ月**
- [ ] ミンコフスキー和・交差計算
- [ ] ボロノイ図・ドロネー三角形分割
- [ ] 高次元対応
- [ ] 機械学習統合例
- [ ] 学術論文執筆

### 品質保証

- **継続的インテグレーション**: GitHub Actions
- **コードカバレッジ**: 90%以上
- **型チェック**: mypy
- **コード品質**: black, flake8, pylint
- **ドキュメント**: sphinx + autodoc

## 競合比較とベンチマーク

### ベンチマーク対象

1. **SciPy** (scipy.spatial.ConvexHull)
2. **Qhull** (直接比較)
3. **CGAL** (C++実装との比較)
4. **Open3D** (3D特化ライブラリ)

### 評価指標

- **計算時間**: 点数・次元数別の処理速度
- **メモリ使用量**: ピークメモリとメモリ効率
- **数値精度**: 既知解との誤差
- **スケーラビリティ**: 大規模データでの性能
- **GPU加速効果**: CPU vs GPU性能比

## リスクと対策

### 技術的リスク

1. **数値安定性**: robust geometric predicatesの実装
2. **高次元での性能**: 次元の呪いへの対策
3. **メモリ不足**: 大規模データでのOOM対策
4. **JAX制約**: JAX特有の制限への対応

### プロジェクトリスク

1. **人的リソース**: 段階的リリースによるリスク分散
2. **競合の出現**: 継続的な機能追加と差別化
3. **コミュニティ形成**: 積極的な情報発信とエンゲージメント

## まとめ

PolytopAXは、JAXエコシステムにおける幾何学計算の新しいスタンダードを目指すプロジェクトです。段階的な開発アプローチにより、リスクを最小化しながら、研究コミュニティと実用分野の両方にとって価値あるツールを提供していきます。