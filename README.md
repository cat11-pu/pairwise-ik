# ik

二维骨链的反向运动学内核：两骨解析解、循环坐标下降与 FABRIK 迭代，只用
Python 标准库，不联网、不装包。

骨链用骨长列表与关节角列表描述：关节角是相对前一根骨的角度，根的位置由
调用方给定，链条的末端点就是根加上各段旋转后的骨向量。

## 目录

- `ik/core.py` —— 全部几何与求解实现
- `tests/test_core.py` —— unittest 用例

## 运行测试

在项目根目录执行：

    python3 -m unittest discover -s tests -v

标准库 Python 3 即可，不需要任何第三方依赖。

## 公开接口

- `two_bone_ik(root, target, l1, l2, bend=1.0, limits=(0.0, PI))`：两骨解析解，
  返回 `TwoBoneSolution(root, mid, end, reachable, interior)`，`interior` 是肘部
  两骨之间的夹角，`bend` 决定肘部偏向哪一侧
- `ccd_solve(origin, angles, lengths, target, limits=None, tol=1e-6, max_iters=24)`：
  循环坐标下降，返回 `CCDResult(angles, points, error, iterations, converged)`
- `fabrik_solve(origin, lengths, target, tol=1e-6, max_iters=16)`：FABRIK 迭代，
  返回 `FABRIKResult(points, error, iterations, converged, reachable)`
- `forward_kinematics(origin, angles, lengths)` 与
  `chain_error(origin, angles, lengths, target)`：正运动学与末端误差
- `wrap_pi(angle)` 与 `clamp_angle(angle, lo, hi)`：角度归位与限位夹紧
