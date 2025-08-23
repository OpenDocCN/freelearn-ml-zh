# 评估

# 第一章，量子计算基础

**(1.1)** 如果量子比特的状态是![1/2 的平方根|0> + 1/2 的平方根|1>](img/2 的平方根|1>"), 测量![0](img/file12.png "0")的概率正好是

![左边的绝对值平方等于 1/2](img/2")

同样，测量![1](img/file13.png "1")的概率也是![1/2](img/2")。如果量子比特的状态是![1/3 的平方根|0> + 2/3 的平方根|1>](img/3 的平方根|1>"), 测量![0](img/file12.png "0")的概率是![左边的 1/3 的平方根的平方等于 1/3](img/3")，测量![1](img/file13.png "1")的概率是![左边的 2/3 的平方根的平方等于 2/3](img/3")

最后，如果量子比特的状态是![1/2 的平方根|0> - 1/2 的平方根|1>](img/2 的平方根|1>"), 测量![0](img/file12.png "0")的概率是![左边的 1/2 的平方根的平方等于 1/2](img/2")，测量![1](img/file13.png "1")的概率是![左边的负 1/2 的平方根的平方等于 1/2](img/2")

**(1.2)** ![\sqrt{\left. 1\slash 2 \right.}\left| 0 \right\rangle + \sqrt{\left. 1\slash 2 \right.}\left| 1 \right\rangle](img/rangle") 和 ![\sqrt{\left. 1\slash 3 \right.}\left| 0 \right\rangle + \sqrt{\left. 2\slash 3 \right.}\left| 1 \right\rangle](img/rangle") 的内积是 ![\sqrt{\left. 1\slash 2 \right.}\sqrt{\left. 1\slash 3 \right.} + \sqrt{\left. 1\slash 2 \right.}\sqrt{\left. 2\slash 3 \right.} = \sqrt{\left. 1\slash 6 \right.} + \sqrt{\left. 1\slash 3 \right.}.](img/right.}.")

![\sqrt{\left. 1\slash 2 \right.}\left| 0 \right\rangle + \sqrt{\left. 1\slash 2 \right.}\left| 1 \right\rangle](img/rangle") 和 ![\sqrt{\left. 1\slash 2 \right.}\left| 0 \right\rangle - \sqrt{\left. 1\slash 2 \right.}\left| 1 \right\rangle](img/rangle") 的内积是 ![\sqrt{\left. 1\slash 2 \right.}\sqrt{\left. 1\slash 2 \right.} - \sqrt{\left. 1\slash 2 \right.}\sqrt{\left. 1\slash 2 \right.} = 0.](img/right.} = 0.")

**(1.3)** ![X](img/file9.png "X") 的伴随是 ![X](img/file9.png "X") 本身，并且满足 ![XX = I](img/file1670.png "XX = I")。因此，![X](img/file9.png "X") 是幺正的，其逆也是 ![X](img/file9.png "X") 本身。操作 ![X](img/file9.png "X") 将 ![a\left| 0 \right\rangle + b\left| 1 \right\rangle](img/rangle") 变为 ![b\left| 0 \right\rangle + a\left| 1 \right\rangle](img/rangle").

**(1.4)** ![H](img/file10.png "H") 的伴随是其自身，并且它满足 ![HH = I](img/file1672.png "HH = I")。因此，![H](img/file10.png "H") 是幺正的，其逆也是 ![H](img/file10.png "H") 本身。操作 ![H](img/file10.png "H") 将 ![\left| + \right\rangle](img/rangle") 映射到 ![\left| 0 \right\rangle](img/rangle")，将 ![\left| - \right\rangle](img/rangle") 映射到 ![\left| 1 \right\rangle](img/rangle")。最后，它还满足 ![X\left| + \right\rangle = \left| + \right\rangle](img/rangle") 和 ![X\left| - \right\rangle = - \left| - \right\rangle](img/rangle").

**(1.5)** 它表明 ![Z\left| 0 \right\rangle = HXH\left| 0 \right\rangle = HX\left| + \right\rangle = H\left| + \right\rangle = \left| 0 \right\rangle](img/rangle") 和 ![Z\left| 1 \right\rangle = HXH\left| 1 \right\rangle = HX\left| - \right\rangle = - H\left| - \right\rangle = - \left| 1 \right\rangle.](img/rangle.") 它还表明 ![\begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & {- \frac{1}{\sqrt{2}}} \\ \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \\ \end{pmatrix}\begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & {- \frac{1}{\sqrt{2}}} \\ \end{pmatrix} = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & {- \frac{1}{\sqrt{2}}} \\ \end{pmatrix}\begin{pmatrix} \frac{1}{\sqrt{2}} & {- \frac{1}{\sqrt{2}}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & {- 1} \\ \end{pmatrix}.](img/end{pmatrix}.")

**(1.6)** 由于![e^{i\frac{\pi}{4}}e^{i\frac{\pi}{4}} = e^{i\frac{\pi}{2}}](img/pi}{2}}")，显然![T^{2} = S](img/file76.png "T^{2} = S")。此外，根据欧拉公式，我们有![e^{i\frac{\pi}{2}}e^{i\frac{\pi}{2}} = e^{i\pi} = - 1](img/pi} = - 1")，因此![S^{2} = Z](img/file78.png "S^{2} = Z")。因此，![SS^{3} = S^{2}S^{2} = ZZ = I](img/file1680.png "SS^{3} = S^{2}S^{2} = ZZ = I")，所以![S^{\dagger} = S^{3}](img/dagger} = S^{3}")。同样，![TT^{7} = T^{2}T^{2}T^{2}T^{2} = S^{4} = I](img/file1682.png "TT^{7} = T^{2}T^{2}T^{2}T^{2} = S^{4} = I")，因此![T^{\dagger} = T^{7}](img/dagger} = T^{7}").

**(1.7)** 通过![R_{X}](img/file118.png "R_{X}")的定义，我们有![R_{X}(\pi) = \begin{pmatrix} {\cos\frac{\pi}{2}} & {- i\sin\frac{\pi}{2}} \\ {- i\sin\frac{\pi}{2}} & {\cos\frac{\pi}{2}} \\ \end{pmatrix} = \begin{pmatrix} 0 & {- i} \\ {- i} & 0 \\ \end{pmatrix} = - iX.](img/pi) = \begin{pmatrix} {\cos\frac{\pi}{2}} & {- i\sin\frac{\pi}{2}} \\ {- i\sin\frac{\pi}{2}} & {\cos\frac{\pi}{2}} \\ \end{pmatrix} = \begin{pmatrix} 0 & {- i} \\ {- i} & 0 \\ \end{pmatrix} = - iX.")

类似地，![R_{Y}(\pi) = \begin{pmatrix} {\cos\frac{\pi}{2}} & {- \sin\frac{\pi}{2}} \\ {\sin\frac{\pi}{2}} & {\cos\frac{\pi}{2}} \\ \end{pmatrix} = \begin{pmatrix} 0 & {- 1} \\ 1 & 0 \\ \end{pmatrix} = - iY](img/pi) = \begin{pmatrix} {\cos\frac{\pi}{2}} & {- \sin\frac{\pi}{2}} \\ {\sin\frac{\pi}{2}} & {\cos\frac{\pi}{2}} \\ \end{pmatrix} = \begin{pmatrix} 0 & {- 1} \\ 1 & 0 \\ \end{pmatrix} = - iY")和![R_{Z}(\pi) = \begin{pmatrix} e^{- i\frac{\pi}{2}} & 0 \\ 0 & e^{i\frac{\pi}{2}} \\ \end{pmatrix} = \begin{pmatrix} {- i} & 0 \\ 0 & i \\ \end{pmatrix} = - iZ.](img/pi) = \begin{pmatrix} e^{- i\frac{\pi}{2}} & 0 \\ 0 & e^{i\frac{\pi}{2}} \\ \end{pmatrix} = \begin{pmatrix} {- i} & 0 \\ 0 & i \\ \end{pmatrix} = - iZ.")

此外，![R_{Z}\left( \frac{\pi}{2} \right) = \begin{pmatrix} e^{- i\frac{\pi}{4}} & 0 \\ 0 & e^{i\frac{\pi}{4}} \\ \end{pmatrix} = e^{- i\frac{\pi}{4}}S](img/right) = \begin{pmatrix} e^{- i\frac{\pi}{4}} & 0 \\ 0 & e^{i\frac{\pi}{4}} \\ \end{pmatrix} = e^{- i\frac{\pi}{4}}S")和![R_{Z}\left( \frac{\pi}{4} \right) = \begin{pmatrix} e^{- i\frac{\pi}{8}} & 0 \\ 0 & e^{i\frac{\pi}{8}} \\ \end{pmatrix} = e^{- i\frac{\pi}{8}}T.](img/right) = \begin{pmatrix} e^{- i\frac{\pi}{8}} & 0 \\ 0 & e^{i\frac{\pi}{8}} \\ \end{pmatrix} = e^{- i\frac{\pi}{8}}T.")

**(1.8)** 从![U(θ,φ,λ)](img/file130.png "U(θ,φ,λ)")的定义出发，我们得到![U(θ,φ,λ)U(θ,φ,λ)† = \begin{pmatrix} \cos\frac{\theta}{2} & - e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\varphi}\sin\frac{\theta}{2} & e^{i{(\varphi + \lambda})}\cos\frac{\theta}{2} \\ \end{pmatrix}\begin{pmatrix} \cos\frac{\theta}{2} & e^{- i\varphi}\sin\frac{\theta}{2} \\ - e^{- i\lambda}\sin\frac{\theta}{2} & e^{- i{(\varphi + \lambda})}\cos\frac{\theta}{2} \\ \end{pmatrix} = I](img/file1689.png "U(θ,φ,λ)U(θ,φ,λ)† = \begin{pmatrix} \cos\frac{\theta}{2} & - e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\varphi}\sin\frac{\theta}{2} & e^{i{(\varphi + \lambda})}\cos\frac{\theta}{2} \\ \end{pmatrix}\begin{pmatrix} \cos\frac{\theta}{2} & e^{- i\varphi}\sin\frac{\theta}{2} \\ - e^{- i\lambda}\sin\frac{\theta}{2} & e^{- i{(\varphi + \lambda})}\cos\frac{\theta}{2} \\ \end{pmatrix} = I")和，类似地，![U(θ,φ,λ)†U(θ,φ,λ) = I](img/file1690.png "U(θ,φ,λ)†U(θ,φ,λ) = I")。因此，![U(θ,φ,λ)](img/file130.png "U(θ,φ,λ)")是正交的。

此外，我们得到![U(θ, - \pi/2,\pi/2) = \begin{pmatrix} \cos\frac{\theta}{2} & - i\sin\frac{\theta}{2} \\ - i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \\ \end{pmatrix} = R_{X}(\theta)](img/2) = \begin{pmatrix} \cos\frac{\theta}{2} & - i\sin\frac{\theta}{2} \\ - i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \\ \end{pmatrix} = R_{X}(\theta). \right.")

类似地，它成立，即![U(θ,0,0) = \begin{pmatrix} \cos\frac{\theta}{2} & - \sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \\ \end{pmatrix} = R_{Y}(\theta)](img/file1692.png "U(θ,0,0) = \begin{pmatrix} \cos\frac{\theta}{2} & - \sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \\ \end{pmatrix} = R_{Y}(\theta)")以及![U(0,0,θ) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \\ \end{pmatrix} = e^{i\frac{\theta}{2}}R_{Z}(\theta)](img/file1693.png "U(0,0,θ) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \\ \end{pmatrix} = e^{i\frac{\theta}{2}}R_{Z}(\theta).")。

**(1.9)** 由于 ![\theta = 2\arccos\sqrt{p}](img/sqrt{p}"), 测量前的状态是 ![\cos\frac{\theta}{2}\left| 0 \right\rangle + \sin\frac{\theta}{2}\left| 1 \right\rangle = \sqrt{p}\left| 0 \right\rangle + \sqrt{1 - p}\left| 1 \right\rangle.](img/rangle.")

因此，测量![0](img/file12.png "0")的概率是![p](img/file141.png "p")，测量![1](img/file13.png "1")的概率是![1 - p](img/file142.png "1 - p").

**(1.10)** 获得符号![1](img/file13.png "1")的概率将是 ![\left| a_{10} \right|^{2} + \left| a_{11} \right|^{2}](img/right|^{2}"). 在这个测量结果下，状态将塌缩到 ![\frac{a_{10}\left| {10} \right\rangle + a_{11}\left| {11} \right\rangle}{\sqrt{\left| a_{10} \right|^{2} + \left| a_{11} \right|^{2}}}.](img/right|^{2}}}.")

**(1.11)** 它表明 ![(U_{1} \otimes U_{2})(U_{1}^{\dagger} \otimes U_{2}^{\dagger}) = (U_{1}U_{1}^{\dagger}) \otimes (U_{2}U_{2}^{\dagger}) = I \otimes I.](img/otimes U_{2})(U_{1}^{\dagger} \otimes U_{2}^{\dagger}) = (U_{1}U_{1}^{\dagger}) \otimes (U_{2}U_{2}^{\dagger}) = I \otimes I.")

类似地，![（\(U_{1}^{\dagger} \otimes U_{2}^{\dagger}\)）（\(U_{1} \otimes U_{2}\)）= \(I \otimes I\)](img/)）（\(U_{1} \otimes U_{2}\)）= \(I \otimes I\")). 因此，\(U_{1} \otimes U_{2}\) 的逆是 \(U_{1}^{\dagger} \otimes U_{2}^{\dagger}\)。另外，根据两个矩阵张量积的定义，对于每一个矩阵 \(A\) 和 \(B\)（即使它们不是幺正的），都有 ![\begin{array}{rlrl} {A^{\dagger} \otimes B^{\dagger} = \begin{pmatrix} a_{11}^{\ast} & a_{21}^{\ast} \\ a_{12}^{\ast} & a_{22}^{\ast} \\ \end{pmatrix} \otimes \begin{pmatrix} b_{11}^{\ast} & b_{21}^{\ast} \\ b_{12}^{\ast} & b_{22}^{\ast} \\ \end{pmatrix}} & {= \begin{pmatrix} {a_{11}^{\ast}\begin{pmatrix} b_{11}^{\ast} & b_{21}^{\ast} \\ b_{12}^{\ast} & b_{22}^{\ast} \\ \end{pmatrix}} & {a_{21}^{\ast}\begin{pmatrix} b_{11}^{\ast} & b_{21}^{\ast} \\ b_{12}^{\ast} & b_{22}^{\ast} \\ \end{pmatrix}} \\ {a_{12}^{\ast}\begin{pmatrix} b_{11}^{\ast} & b_{21}^{\ast} \\ b_{12}^{\ast} & b_{22}^{\ast} \\ \end{pmatrix}} & {a_{22}^{\ast}\begin{pmatrix} b_{11}^{\ast} & b_{21}^{\ast} \\ b_{12}^{\ast} & b_{22}^{\ast} \\ \end{pmatrix}} \\ \end{pmatrix}\qquad} & & \qquad \\ & {= \begin{pmatrix} {a_{11}^{\ast}b_{11}^{\ast}} & {a_{11}^{\ast}b_{21}^{\ast}} & {a_{21}^{\ast}b_{11}^{\ast}} & {a_{21}^{\ast}b_{21}^{\ast}} \\ {a_{11}^{\ast}b_{12}^{\ast}} & {a_{11}^{\ast}b_{22}^{\ast}} & {a_{21}^{\ast}b_{12}^{\ast}} & {a_{21}^{\ast}b_{22}^{\ast}} \\ {a_{12}^{\ast}b_{11}^{\ast}} & {a_{12}^{\ast}b_{21}^{\ast}} & {a_{22}^{\ast}b_{11}^{\ast}} & {a_{22}^{\ast}b_{21}^{\ast}} \\ {a_{12}^{\ast}b_{12}^{\ast}} & {a_{12}^{\ast}b_{22}^{\ast}} & {a_{22}^{\ast}b_{12}^{\ast}} & {a_{22}^{\ast}b_{22}^{\ast}} \\ \end{pmatrix} = {(A \otimes B)}^{\dagger}.\qquad} & & \qquad \\ \end{array}](img/otimes B)}^{\dagger}.\qquad} & & \qquad \\ \end{array}")

**(1.12)** ![X \otimes X](img/otimes X") 的矩阵是 ![\begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ \end{pmatrix}.](img/end{pmatrix}.")

![H \otimes I](img/otimes I") 的矩阵是 ![\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & {- 1} & 0 \\ 0 & 1 & 0 & {- 1} \\ \end{pmatrix}.](img/end{pmatrix}.")

**(1.13)** 在电路

![HHHH ](img/file1702.jpg)

状态 ![\left| {00} \right\rangle](img/rangle") 和 ![\left| {10} \right\rangle](img/rangle") 保持不变，而 ![\left| {01} \right\rangle](img/rangle") 和 ![\left| {11} \right\rangle](img/rangle") 被映射到对方。这正是控制位在底部、目标位在顶部的 CNOT 门的作用。

电路的矩阵是

![\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ \end{pmatrix},](img/end{pmatrix},")

这正是从底部量子位到顶部量子位的 CNOT 门的矩阵。

另一方面，电路

![ ](img/file1704.jpg)

保持 ![\left| {00} \right\rangle](img/rangle") 和 ![\left| {11} \right\rangle](img/rangle") 不变，同时将 ![\left| {01} \right\rangle](img/rangle") 和 ![\left| {10} \right\rangle](img/rangle") 映射到对方。这正是 SWAP 门的作用。

或者，电路的矩阵是

![\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ \end{pmatrix},](img/end{pmatrix},")

这又是 SWAP 门的矩阵。

**(1.14)** 状态 ![\sqrt{\left. 1\slash 3 \right.}(\left| {00} \right\rangle + \left| {01} \right\rangle + \left| {11} \right\rangle)](img/rangle)") 确实是纠缠的。然而，![\frac{1}{2}(\left| {00} \right\rangle + \left| {01} \right\rangle + \left| {10} \right\rangle + \left| {11} \right\rangle)](img/rangle)") 是一个乘积态，因为它可以写成 ![\left| + \right\rangle\left| + \right\rangle.](img/rangle.")

**(1.15)** 如果 ![U](img/file51.png "U") 的矩阵为 ![{(u_{ij})}_{i,j = 1}^{2}](img/file231.png "{(u_{ij})}_{i,j = 1}^{2}"), 则 ![|00>](img/file198.png "|00>") 和 ![|01>](img/file199.png "|01>") 由 ![C_U](img/file229.png "C_U") 保持不变。此外，![|10>](img/file200.png "|10>") 被转换为 ![|1>(u_{11}|0> + u_{21}|1>)](img/file1707.png "|1>(u_{11}|0> + u_{21}|1>"))，而 ![|11>](img/file201.png "|11>") 被转换为 ![|1>(u_{12}|0> + u_{22}|1>)](img/file1708.png "|1>(u_{12}|0> + u_{22}|1>))。因此，![C_U](img/file229.png "C_U") 的矩阵是

![\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & u_{11} & u_{12} \\ 0 & 0 & u_{21} & u_{22} \\ \end{pmatrix}.](img/end{pmatrix}.")

![C_U](img/file229.png "C_U") 的伴随矩阵是 ![C_U^†](img/file1710.png "C_U^†")，并且满足 ![C_U C_U^† = C_U^† C_U = I](img/file1711.png "C_U C_U^† = C_U^† C_U = I")。因此，![C_U](img/file229.png "C_U") 是幺正的。

**(1.16)** 等价性直接来源于 ![HXH = Z](img/file1712.png "HXH = Z") 的事实。

**(1.17)** 我们可以使用以下电路来制备 ![√(1/2)(|00> - |11>)](img/2)(|00> - |11>))：

![|H|Z00⟩⟩](img/file1713.jpg)

我们可以使用电路

![|0HX|0⟩⟩](img/file1714.jpg)

用于制备 ![√(1/2)(|10> + |01>)](img/2)(|10> + |01>))。

最后，电路

![|0HX|0Z⟩⟩](img/file1716.jpg)

可以用来获得 ![√(1/2)(|10> - |01>)](img/2)(|10> - |01>))。

注意，为了制备这些状态，我们只使用了附加到电路中的张量积门，这些电路用于获得原始贝尔态 ![√(1/2)(|00> + |11>)](img/2)(|00> + |11>))。例如，我们有

![\sqrt{\left. 1\slash 2 \right.}(\left| {10} \right\rangle - \left| {01} \right\rangle) = (X \otimes Z)\sqrt{\left. 1\slash 2 \right.}(\left| {00} \right\rangle + \left| {11} \right\rangle)](img/rangle) = (X \otimes Z)\sqrt{\left. 1\slash 2 \right.}(\left| {00} \right\rangle + \left| {11} \right\rangle)")

然后，这也成立：

![(X \otimes Z)\sqrt{\left. 1\slash 2 \right.}(\left| {10} \right\rangle - \left| {01} \right\rangle) = \sqrt{\left. 1\slash 2 \right.}(\left| {00} \right\rangle + \left| {11} \right\rangle).](img/otimes Z)\sqrt{\left. 1\slash 2 \right.}(\left| {10} \right\rangle - \left| {01} \right\rangle) = \sqrt{\left. 1\slash 2 \right.}(\left| {00} \right\rangle + \left| {11} \right\rangle).")

如果 ![\sqrt{\left. 1\slash 2 \right.}(\left| {10} \right\rangle - \left| {01} \right\rangle)](img/rangle)") 是一个乘积态，那么 ![\sqrt{\left. 1\slash 2 \right.}(\left| {00} \right\rangle + \left| {11} \right\rangle)](img/rangle)") 也将是一个乘积态。但是这是不可能的，因为我们知道 ![\sqrt{\left. 1\slash 2 \right.}(\left| {00} \right\rangle + \left| {11} \right\rangle)](img/rangle)") 是纠缠态的。

**(1.18)** 我们可以通过归纳法来证明它。我们知道当 ![n = 1](img/file1720.png "n = 1") 时结果是正确的。现在，假设当 ![n > 1](img/file1721.png "n > 1") 时它是正确的，并考虑一个 ![n + 1](img/file792.png "n + 1") 个量子比特的基态 ![\left| \psi \right\rangle](img/rangle")。如果 ![\left| \psi \right\rangle = \left| 0 \right\rangle\left| \psi^{\prime} \right\rangle](img/rangle")，那么 ![\left| \psi \right\rangle](img/rangle") 的列向量将始于 ![\left| \psi^{\prime} \right\rangle](img/rangle") 的列向量的元素，然后它将有 ![2^{n}](img/file256.png "2^{n}") 个零。但是，根据归纳假设，![\left| \psi^{\prime} \right\rangle](img/rangle") 的列向量正好是我们感兴趣的形式。因此，![\left| \psi \right\rangle](img/rangle") 也具有所需的结构。当 ![\left| \psi \right\rangle = \left| 1 \right\rangle\left| \psi^{\prime} \right\rangle](img/rangle") 时情况类似。

另一方面，由于每个 ![n](img/file244.png "n")-量子比特状态都可以写成基态的归一化线性组合，因此它的向量表示是一个具有 ![2^{n}](img/file256.png "2^{n}") 个坐标的单位长度列向量。

**(1.19)** 如果我们测量一个通用多量子比特状态的 ![j](img/file258.png "j")-量子比特，得到 ![1](img/file13.png "1") 的概率由以下公式给出

![\sum\limits_{l \in J_{1}}\left| a_{l} \right|^{2},](img/right|^{2},")

其中 ![J_{1}](img/file1726.png "J_{1}") 是那些 ![j](img/file258.png "j")-位为 ![1](img/file13.png "1") 的数字集合。坍缩后的状态将是

![\frac{\sum\limits_{l \in J_{1}}a_{l}\left| l \right\rangle}{\sqrt{\sum\limits_{l \in J_{1}}\left| a_{i} \right|^{2}}}.](img/right|^{2}}}.")

**(1.20)** 当我们测量 ![\left. (1\slash 2)\left| {100} \right\rangle + (1\slash 2)\left| {010} \right\rangle + \sqrt{\left. 1\slash 2 \right.}\left| {001} \right\rangle \right.\left| {100} \right\rangle + (1\slash 2)\left| {010} \right\rangle + \sqrt{\left. 1\slash 2 \right.}\left| {001} \right\rangle \right.") 的第二个量子比特时，得到 ![0](img/file12.png "0") 的概率是

![\left| \frac{1}{2} \right|^{2} + \left| \frac{1}{\sqrt{2}} \right|^{2} = \frac{1}{4} + \frac{1}{2} = \frac{3}{4}.](img/frac{3}{4}.")

测量第二个量子比特并得到 ![0](img/file12.png "0") 的结果将是

![\frac{1}{\sqrt{3}}\left| {100} \right\rangle + \frac{\sqrt{2}}{\sqrt{3}}\left| {001} \right\rangle.](img/rangle.")

**(1.21)** 让我们表示 ![x = x_{1}\ldots x_{n}](img/ldots x_{n}") 和 ![y = y_{1}\ldots y_{n}](img/ldots y_{n}"), 其中 ![x_{i}](img/file714.png "x_{i}") 是 ![x](img/file269.png "x") 的 ![i](img/file49.png "i")-位，而 ![y_{i}](img/file1732.png "y_{i}") 是 ![y](img/file270.png "y") 的 ![i](img/file49.png "i")-位。那么，它成立：

![\left\langle y \middle| x \right\rangle = \left\langle y_{1} \middle| x_{1} \right\rangle\ldots\left\langle y_{n} \middle| x_{n} \right\rangle.](img/rangle.")

因此，当![x = y](img/file941.png "x = y")时，![\left\langle y \middle| x \right\rangle = 1](img/rangle = 1")；而当![x \neq y](img/neq y")时，![\left\langle y \middle| x \right\rangle = 0](img/rangle = 0")。由此可知，![{\{\left| x \right\rangle\}}_{x \in {\{ 0,1\}}^{n}}](img/}}^{n}}")中的元素是正交归一的。由于这个集合的基数是![2^{n}](img/file256.png "2^{n}")，它是![n](img/file244.png "n")-量子比特状态的维度，我们可以得出结论，这个集合构成一个基。

**(1.22)** 成立的是

![\begin{array}{rlrl} & {\frac{1}{\sqrt{2}}\left( {\left\langle {000} \right| + \left\langle {111} \right|} \right)\frac{1}{2}\left( {\left| {000} \right\rangle + \left| {011} \right\rangle + \left| {101} \right\rangle + \left| {110} \right\rangle} \right)\qquad} & & \qquad \\ & {\qquad = \frac{1}{2\sqrt{2}}(\left\langle 000 \middle| 000 \right\rangle + \left\langle 000 \middle| 011 \right\rangle + \left\langle 000 \middle| 101 \right\rangle + \left\langle 000 \middle| 110 \right\rangle + \qquad} & & \qquad \\ & {\qquad\qquad\left\langle 111 \middle| 000 \right\rangle + \left\langle 111 \middle| 011 \right\rangle + \left\langle 111 \middle| 101 \right\rangle + \left\langle 111 \middle| 110 \right\rangle)\qquad} & & \qquad \\ & {\qquad = \frac{1}{2\sqrt{2}},\qquad} & & \qquad \\ \end{array}](img/right)\frac{1}{2}\left( {\left| {000} \right\rangle + \left| {011} \right\rangle + \left| {101} \right\rangle + \left| {110} \right\rangle} \right)\qquad} & & \qquad \\  & {\qquad = \frac{1}{2\sqrt{2}}(\left\langle 000 \middle| 000 \right\rangle + \left\langle 000 \middle| 011 \right\rangle + \left\langle 000 \middle| 101 \right\rangle + \left\langle 000 \middle| 110 \right\rangle + \qquad} & & \qquad \\  & {\qquad\qquad\left\langle 111 \middle| 000 \right\rangle + \left\langle 111 \middle| 011 \right\rangle + \left\langle 111 \middle| 101 \right\rangle + \left\langle 111 \middle| 110 \right\rangle)\qquad} & & \qquad \\  & {\qquad = \frac{1}{2\sqrt{2}},\qquad} & & \qquad \\ \end{array}")

因为所有内积都是![0](img/file12.png "0")，除了![\left\langle 000 \middle| 000 \right\rangle](img/rangle")，它是![1](img/file13.png "1")。

**(1.23)** 从其作用于基态的作用中，我们推断出 CCNOT 门的矩阵为：

![矩阵](img/file1737.png "矩阵")

该矩阵是其自身的伴随矩阵，并且其平方是单位矩阵。因此，该矩阵是正交的。

**(1.24**) 该电路

![HTTTTHTTT††† ](img/file1738.jpg)

保持所有状态不变，除了![左| {011}右>](img/file1739.png "左| {011}右>")和![左| {111}右>](img/file1740.png "左| {111}右>")。它还交换![左| {011}右>](img/file1739.png "左| {011}右>")和![左| {111}右>](img/file1740.png "左| {111}右>")。这正是目标在顶层量子比特上的 CCNOT 门的作用。

# 第二章，量子计算中的工具

**(2.1**) 我们已经在*附录**D*，*安装工具*中给出了解决方案。

**(2.2**) 为了构建图*2.2b*中的电路，你需要执行以下代码片段：

```py

from qiskit import * 

import numpy as np 

qc = QuantumCircuit(2) 

qc.z(0) 

qc.y(1) 

qc.cry(np.pi/2, 0, 1) 

qc.u(np.pi/4, np.pi, 0, 0) 

qc.rz(np.pi/4,1)

```

如果你想要可视化电路，当然，你可以使用`qc.draw("mpl")`。

**(2.3**) 你可以检查 IBM 自己的方法实现([`github.com/Qiskit/qiskit-terra/blob/5ccf3a41cb10742ae2158b6ee9d13bbb05f64f36/qiskit/circuit/quantumcircuit.py#L2205`](https://github.com/Qiskit/qiskit-terra/blob/5ccf3a41cb10742ae2158b6ee9d13bbb05f64f36/qiskit/circuit/quantumcircuit.py#L2205))并与你自己的进行比较！

它们采取了我们没有考虑的一些额外步骤，例如在电路中添加**障碍**，但你可以忽略这些细节。

**(2.4**) 你已经在*附录**D*，*安装工具*中找到了解决方案。

**(2.5**) 我们已经看到了如何在 Qiskit 中构建这些电路。要在 PennyLane 中构建它们，我们需要运行以下代码片段：

```py

import pennylane as qml 

import numpy as np 

dev = qml.device(’default.qubit’, wires = 2) 

@qml.qnode(dev) 

def qcircA(): 

    qml.PauliX(wires = 0) 

    qml.RX(np.pi/4, wires = 1) 

    qml.CNOT(wires = [0,1]) 

    qml.U3(np.pi/3, 0, np.pi, wires = 0) 

    return qml.state() 

@qml.qnode(dev) 

def qcircB(): 

    qml.PauliZ(wires = 0) 

    qml.PauliY(wires = 1) 

    qml.CRY(np.pi/2, wires = [0,1]) 

    qml.U3(np.pi/4, np.pi, 0, wires = 0) 

    qml.RZ(np.pi/4, wires = 1) 

    return qml.state()

```

如果我们执行`print(qcircB())`来运行电路 B，我们得到以下状态向量：

```py

tensor([ 0\.        +0.j        , -0.35355339+0.85355339j, 
         0\.        +0.j        ,  0.14644661-0.35355339j], 
         requires_grad=True)

```

另一方面，如果我们使用 Qiskit 模拟相同的电路，我们得到以下输出：

```py

Statevector([-5.65831421e-17-3.20736464e-17j, 
              2.34375049e-17+1.32853393e-17j, 
             -3.53553391e-01+8.53553391e-01j, 
              1.46446609e-01-3.53553391e-01j], 
            dims=(2, 2))

```

注意，这与我们用 PennyLane 得到的结果相同。首先，我们必须考虑到前两个值——从计算的角度来看——是零。然后，我们必须关注 Qiskit 如何根据其自己的约定，以下列顺序给出基态的振幅：![\left| {00} \right\rangle](img/rangle")、![\left| {10} \right\rangle](img/rangle")、![\left| {01} \right\rangle](img/rangle") 和 ![\left| {11} \right\rangle](img/rangle")。

# 第三章，处理二次无约束二进制优化问题

**(3.1)** 我们可以将顶点 ![0](img/file12.png "0")、![1](img/file13.png "1") 和 ![4](img/file143.png "4") 放在同一组，将顶点 ![2](img/file302.png "2") 和 ![3](img/file472.png "3") 放在另一组。然后，五条边属于该割集，即 ![(0,2),(1,2),(1,3),(2,4)](img/file1741.png "(0,2),(1,2),(1,3),(2,4)") 和 ![(3,4)](img/file1742.png "(3,4)").

**(3.2)** *图* **3.3* 中图的 Max-Cut 优化问题是

*![\begin{array}{rlrl} {\text{Minimize~}\quad} & {z_{0}z_{1} + z_{0}z_{2} + z_{1}z_{2} + z_{1}z_{4} + z_{2}z_{3} + z_{3}z_{4} + z_{3}z_{5} + z_{4}z_{5}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,\ldots,5.\qquad} & & \qquad \\ \end{array}](img/end{array}")

给定 ![z_{0} = z_{1} = z_{2} = 1](img/file334.png "z_{0} = z_{1} = z_{2} = 1") 和 ![z_{3} = z_{4} = z_{5} = - 1](img/file335.png "z_{3} = z_{4} = z_{5} = - 1") 的割集值为 ![4](img/file143.png "4")。这个割集不是最优的，因为例如，![z_{0} = z_{1} = z_{2} = z_{5} = 1](img/file1744.png "z_{0} = z_{1} = z_{2} = z_{5} = 1") 和 ![z_{3} = z_{4} = - 1](img/file1745.png "z_{3} = z_{4} = - 1") 可以得到更低的值。

**(3.3)** 成立的是 ![\left\langle {010} \right|\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \right)\left| {010} \right\rangle = 0](img/right)\left| {010} \right\rangle = 0") 和 ![\left\langle {100} \right|\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \right)\left| {100} \right\rangle = - 2](img/right)\left| {100} \right\rangle = - 2")。这个值是最小可能的，因为我们图中只有两条边。

**(3.4)** 我们可以使用以下代码计算所需的期望值：

```py

from qiskit.quantum_info import Pauli 

from qiskit.opflow.primitive_ops import PauliOp 

from qiskit.quantum_info import Statevector 

H_cut = PauliOp(Pauli("ZZI")) + PauliOp(Pauli("ZIZ")) 

for x in range(8): # We consider x=0,1...7 

    psi = Statevector.from_int(x, dims = 8) 

    print("The expectation value of |",x,">", "is", 

        psi.expectation_value(H_cut))

```

如果我们运行它，我们将得到以下输出：

```py

The expectation value of | 0 > is (2+0j) 

The expectation value of | 1 > is 0j 

The expectation value of | 2 > is 0j 

The expectation value of | 3 > is (-2+0j) 

The expectation value of | 4 > is (-2+0j) 

The expectation value of | 5 > is 0j 

The expectation value of | 6 > is 0j 

The expectation value of | 7 > is (2+0j)

```

因此，我们可以看到有两个状态可以获得最优值，并且它们都对应于![0](img/file12.png "0")在一组中，而![1](img/file13.png "1")和![2](img/file302.png "2")在另一组中的割。

**(3.5)** QUBO 问题将是

![\begin{array}{rlrl} {\text{最小化~}\quad} & {x_{0}^{2} - 4x_{0}x_{1} + 6x_{0}x_{2} - 8x_{0}x_{3} + 4x_{1}^{2} - 12x_{1}x_{2} + 16x_{1}x_{3} + 9x_{2}^{2}\qquad} & & \qquad \\ & {- 24x_{2}x_{3} + 16x_{3}^{2}\qquad} & & \qquad \\ {\text{约束条件~}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2,3.\qquad} & & \qquad \\ \end{array}](img/end{array}")

等价的 Ising 基态问题将是

![\begin{array}{rlrl} {\text{最小化~}\quad} & {- z_{0}z_{1} + \frac{3z_{0}z_{2}}{2} - 2z_{0}z_{3} + z_{0} - 3z_{1}z_{2} + 4z_{1}z_{3} - 2z_{1} - 6z_{2}z_{3} + 3z_{2} - 4z_{3}\qquad} & & \qquad \\ {\text{约束条件~}\quad} & {z_{j} \in \{ 1, - 1\},\qquad j = 0,1,2,3,\qquad} & & \qquad \\ \end{array}](img/end{array}")

其中我们省略了独立项![\frac{17}{2}](img/frac{17}{2}").

**(3.6)** 二进制线性规划将是

![\begin{array}{rlrl} {\text{最小化}\quad} & {- 3x_{0} - x_{1} - 7x_{2} - 7x_{3}\qquad} & & \qquad \\ {\text{约束条件}\quad} & {2x_{0} + x_{1} + 5x_{2} + 4x_{3} \leq 8,\qquad} & & \qquad \\ & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2,3.\qquad} & & \qquad \\ \end{array}](img/end{array}")

**(3.7)** QUBO 问题如下

![优化表达式](img/file1752.png "优化表达式")

**(3.8)** 路径成本的表示式是

![矩阵表达式](img/file1753.png "矩阵表达式")

# 第四章，绝热量子计算和量子退火

**(4.1)** 我们首先考虑一个状态 ![j](img/rangle")，其中每个 ![j](img/rangle") 要么是 ![j](img/rangle")，要么是 ![j](img/rangle")。对于正交基的所有 ![j](img/file256.png "2^{n}") 这样的状态，以及因此任何一般的 ![j](img/rangle") 状态，可以写成 ![j](img/rangle,") 其中 ![j](img/right.")。

然后，对于每个 ![j](img/file258.png "j")，它满足

![\left\langle x \right|X_{j}\left| x \right\rangle = \left\langle x_{j} \right|X_{j}\left| x_{j} \right\rangle.](img/rangle.")

但 ![j](img/rangle") 当 ![j](img/file13.png "1") 时，如果 ![j](img/rangle")，而当 ![j](img/file312.png "- 1") 时，如果 ![j](img/rangle"). 因此，它满足 ![j](img/right.") 因为 ![j](img/right.") 对于每个 ![j](img/file269.png "x") 和 ![j](img/right.").

然后，由于 ![H_{0} = - {\sum}_{j = 0}^{n - 1}X_{j}](img/sum}_{j = 0}^{n - 1}X_{j}"), 根据线性性质，我们得到 ![\left\langle \psi \right|H_{0}\left| \psi \right\rangle = - \sum\limits_{j = 0}^{n - 1}\left\langle \psi \right|X_{j}\left| \psi \right\rangle \geq - n.](img/geq - n.")

另一方面，如果我们考虑 ![\left| \psi_{0} \right\rangle = {\otimes}_{i = 0}^{n - 1}\left| + \right\rangle](img/rangle")，根据之前的推理，我们有 ![\left\langle \psi_{0} \right|X_{j}\left| \psi_{0} \right\rangle = 1](img/rangle = 1")。因此， ![\left\langle \psi_{0} \right|H_{0}\left| \psi_{0} \right\rangle = - n](img/rangle = - n")，这是可能的最小值，因此 ![\left| \psi_{0} \right\rangle](img/rangle") 是我们寻找的基态。

**(4.2)** 我们可以使用以下代码定义最小化 ![x_{0}x_{2} - x_{0}x_{1} + 2x_{1}](img/file1766.png "x_{0}x_{2} - x_{0}x_{1} + 2x_{1}") 的 QUBO 问题：

```py

import dimod 

J = {(0,1):-1, (0,2):1} 

h = {1:2} 

problem = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.BINARY) 

print("The problem we are going to solve is:") 

print(problem)

```

我们可以使用以下方法来解决它：

```py

from dwave.system import DWaveSampler 

from dwave.system import EmbeddingComposite 

sampler = EmbeddingComposite(DWaveSampler()) 

result = sampler.sample(problem, num_reads=10) 

print("The solutions that we have obtained are") 

print(result)

```

**(4.3)** 为了简单起见，我们将松弛变量表示为 ![s_{0}](img/file1767.png "s_{0}") 和 ![s_{1}](img/file1768.png "s_{1}")。然后，惩罚项是 ![{(y_{0} + 2y_{1} + s_{0} + s_{1} - 2)}^{2}](img/file1769.png "{(y_{0} + 2y_{1} + s_{0} + s_{1} - 2)}^{2}")。当您将其乘以 5，展开并添加到成本函数中时，您将获得 `cqm_to_bqm` 方法计算的确切表达式。

**(4.4)** 对于 ![0](img/file12.png "0") 到 ![7](img/file465.png "7") 的量子位，我们有以下连接：

```py

{0: {4, 5, 6, 7, 128}, 1: {4, 5, 6, 7, 129}, 

 2: {4, 5, 6, 7, 130}, 3: {4, 5, 6, 7, 131}, 

 4: {0, 1, 2, 3, 12}, 5: {0, 1, 2, 3, 13}, 

 6: {0, 1, 2, 3, 14}, 7: {0, 1, 2, 3, 15}}

```

显然，从 ![0](img/file12.png "0") 到 ![3](img/file472.png "3") 的每个顶点都连接到从 ![4](img/file143.png "4") 到 ![7](img/file465.png "7") 的每个顶点，正如我们所需要的。此外，从 ![0](img/file12.png "0") 到 ![3](img/file472.png "3") 的每个顶点都连接到位于第一个单元格下面的 ![128](img/file563.png "128") 到 ![131](img/file1770.png "131") 的一个顶点，而从 ![4](img/file143.png "4") 到 ![7](img/file465.png "7") 的每个顶点都连接到位于第一个单元格右侧的 ![12](img/file601.png "12") 到 ![15](img/file599.png "15") 的一个顶点。

**(4.5)** 您可以使用以下说明轻松检查这些值：

```py

sampler = DWaveSampler(solver = "DW_2000Q_6") 

print("The default annealing time is", 

    sampler.properties["default_annealing_time"],"microsends") 

print("The possible values for the annealing time (in microseconds)"\ 

    " lie in the range",sampler.properties["annealing_time_range"])

```

在这种情况下，输出将如下所示：

```py

The default annealing time is 20.0 microsends 

The possible values for the annealing time (in microseconds) 

    lie in the range [1.0, 2000.0]

```

# 第五章，QAOA：量子近似优化算法

**(5.1)** 对于 ![Z_{1}Z_{3} + Z_{0}Z_{2} - 2Z_{1} + 3Z_{2}](img/file689.png "Z_{1}Z_{3} + Z_{0}Z_{2} - 2Z_{1} + 3Z_{2}")，其中 ![p = 1](img/file676.png "p = 1") 的 QAOA 电路如下：

![HRHRRHRRRHRR ((−((2(6((2(22γγ2γ2β4βββγ)))))))) XZXZZXZX 11111111 ](img/file1771.jpg)

**(5.2)** 它满足以下条件：

![\left\langle {100} \right|H_{1}\left| {100} \right\rangle = 3\left\langle {100} \right|Z_{0}Z_{2}\left| {100} \right\rangle - \left\langle {100} \right|Z_{1}Z_{2}\left| {100} \right\rangle + 2\left\langle {100} \right|Z_{0}\left| {100} \right\rangle = - 3 - 1 - 2 = - 6.](img/rangle = - 3 - 1 - 2 = - 6.")

**(5.3)** 我们可以将问题重新表述为

![开始{array}{rlrl} {最小化~} & {(1 - x_{0})(1 - x_{1})x_{2}(1 - x_{3}) + x_{0}(1 - x_{1})(1 - x_{2})(1 - x_{3}) + x_{0}(1 - x_{1})x_{2}x_{3}} & & & & \\ {subject~to~} & {x_{j} \in \{ 0,1\},\quad j = 0,1,2,3.} & & & & \\ & & & & \\ \end{array}](img/begin{array}{rlrl} {最小化~} & {(1 - x_{0})(1 - x_{1})x_{2}(1 - x_{3}) + x_{0}(1 - x_{1})(1 - x_{2})(1 - x_{3}) + x_{0}(1 - x_{1})x_{2}x_{3}} & & & & \\ {subject~to~} & {x_{j} \in \{ 0,1\},\quad j = 0,1,2,3.} & & & & \\ & & & & \\ \end{array}")

**(5.4)** 该操作可以用以下电路实现：

![|||||RxxxxxZ01234⟩⟩⟩⟩⟩(π2) ](img/file1774.jpg)

**(5.5)** 它认为

![开始{array}{rlrl} {左 langle {100} |H_{1}左| {100} rangle} & {= 左 langle {100} |Z_{0}Z_{1}Z_{2}左| {100} rangle + 3 左 langle {100} |Z_{0}Z_{2}左| {100} rangle - 左 langle {100} |Z_{1}Z_{2}左| {100} rangle} & & & & \\ & {+ 2 左 langle {100} |Z_{0}左| {100} rangle = - 1 - 3 - 1 - 2 = - 7.} & & & & \\ \end{array}](img/end{array}")

**(5.6)** 您可以使用以下代码获得可重复的结果：

```py

from qiskit import Aer 

from qiskit.algorithms import QAOA 

from qiskit.algorithms.optimizers import COBYLA 

from qiskit.utils import algorithm_globals, QuantumInstance 

from qiskit_optimization.algorithms import MinimumEigenOptimizer 

seed = 1234 

algorithm_globals.random_seed = seed 

quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), 

    shots = 1024, seed_simulator=seed, seed_transpiler=seed) 

qaoa = QAOA(optimizer = COBYLA(), 

            quantum_instance=quantum_instance, reps = 1) 

qaoa_optimizer = MinimumEigenOptimizer(qaoa) 

result = qaoa_optimizer.solve(qp) 

print(’Variable order:’, [var.name for var in result.variables]) 

for s in result.samples: 

    print(s)

```

**(5.7)** 我们可以使用以下说明定义![- 3Z_{0}Z_{1}Z_{2} + 2Z_{1}Z_{2} - Z_{2}](img/file773.png "- 3Z_{0}Z_{1}Z_{2} + 2Z_{1}Z_{2} - Z_{2}")哈密顿量：

```py

coefficients = [-3,2,-1] 

paulis = [PauliZ(0)@PauliZ(1)@PauliZ(2), 

    PauliZ(1)@PauliZ(2),PauliZ(2)] 

H = qml.Hamiltonian(coefficients,paulis)

```

我们也可以使用

```py

H = -3*PauliZ(0)@PauliZ(1)@PauliZ(2) 

    + 2*PauliZ(1)@PauliZ(2) -PauliZ(2)

```

# 第六章，GAS：Grover 自适应搜索

**(6.1)** 根据我们的定义，![O_{f}](img/file791.png "O_{f}")总是从一个基态转换到另一个基态。因此，在矩阵表示中，其列向量中恰好有一个元素是![1](img/file13.png "1")，其余都是![0](img/file12.png "0")。这意味着，特别是，所有它的项都是实数。

此外，这个矩阵是对称的。为了证明这一点，假设矩阵有![m_{jk}](img/file1776.png "m_{jk}")这样的元素。如果它不是对称的，那么存在![j,k](img/file1777.png "j,k")使得![m_{jk} \neq m_{kj}](img/neq m_{kj}")。我们可以假设，不失一般性，![m_{jk} = 0](img/file1779.png "m_{jk} = 0")和![m_{kj} = 1](img/file1780.png "m_{kj} = 1")。我们还知道![O_{f}O_{f} = I](img/file1781.png "O_{f}O_{f} = I")，所以矩阵的平方是单位矩阵。特别是，因为这是矩阵平方中行![j](img/file258.png "j")，列![j](img/file258.png "j")的元素，所以![{\sum}_{l}m_{jl}m_{lj} = 1](img/sum}_{l}m_{jl}m_{lj} = 1")。但我们知道![m_{jk}m_{kj} = 0 \cdot 1 = 0](img/cdot 1 = 0")，并且如果![l \neq k](img/neq k")，那么![m_{lj} = 0](img/file1784.png "m_{lj} = 0")，因为每一列中只有一个![1](img/file13.png "1")。然而，那么，![{\sum}_{l}m_{jl}m_{lj} = 0](img/sum}_{l}m_{jl}m_{lj} = 0")，这是矛盾的。

因此，我们有![O_{f}^{\dagger} = O_{f}](img/dagger} = O_{f}")，由于![O_{f}O_{f} = I](img/file1781.png "O_{f}O_{f} = I")，所以![O_{f}](img/file791.png "O_{f}")是幺正的。

**(6.2)** 我们可以使用以下电路：

![XXXXXXXX ](img/file1788.jpg)

**(6.3)** ![10](img/file161.png "10")的表示是![01010](img/file1789.png "01010")，而![−7](img/file470.png "−7")的表示是![11001](img/file1790.png "11001")。它们的和是![00011](img/file1791.png "00011")，编码了![3](img/file472.png "3")。

**(6.4)** 我们可以使用以下电路：

![HPPHPPHPPHPP((((((((6−6−6−6−ππππ)4444π)π)π)π)))) 248248 ](img/file1792.jpg)

**(6.5)** 我们可以使用以下电路：

![ ππππ ππ |x|x|x|0HPPP|0HPPP|0HPPP012⟩(((⟩(((⟩(((⟩⟩⟩2−π2−22−4π)2)4))3)3)3π24))) ](img/file1793.jpg)

**(6.6)** 我们可以使用以下代码：

```py

from qiskit_optimization.problems import QuadraticProgram 

from qiskit_optimization.algorithms import GroverOptimizer 

from qiskit import Aer 

from qiskit.utils import algorithm_globals, QuantumInstance 

seed = 1234 

algorithm_globals.random_seed = seed 

qp = QuadraticProgram() 

qp.binary_var(’x’) 

qp.binary_var(’y’) 

qp.binary_var(’z’) 

qp.minimize(linear = {’x’:3,’y’:2,’z’:-3}, quadratic = {(’x’,’y’):3}) 

quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), 

    shots = 1024, seed_simulator = seed, seed_transpiler=seed) 

grover_optimizer = GroverOptimizer(num_value_qubits = 5, 

    num_iterations=4, quantum_instance=quantum_instance) 

results = grover_optimizer.solve(qp) 

print(results)

```

# 第七章，变分量子本征求解器（VQE）

**(7.1)** 所有这些都源于矩阵是对角线且所有对角线元素都不同的事实，其特征值对应于测量结果的实际标签。

记住，如果一个算符相对于基的坐标矩阵是对角的，这意味着基向量是该算符的特征向量，而且更重要的是，相应的特征值位于对角线上。

**(7.2)** 我们知道

![\left( {A_{1} \otimes \cdots \otimes A_{n}} \right)\left| \lambda_{1} \right\rangle \otimes \cdots \otimes \left| \lambda_{n} \right\rangle = A_{1}\left| \lambda_{1} \right\rangle \otimes \cdots \otimes A_{n}\left| \lambda_{n} \right\rangle.](img/right)\left| \lambda_{1} \right\rangle \otimes \cdots \otimes \left| \lambda_{n} \right\rangle = A_{1}\left| \lambda_{1} \right\rangle \otimes \cdots \otimes A_{n}\left| \lambda_{n} \right\rangle.") 由于 ![A_{j}\left| \lambda_{j} \right\rangle = \lambda_{j}\left| \lambda_{j} \right\rangle](img/rangle"), 结果直接得出。

**(7.3)** 它包含以下内容：![Z\left| 0 \right\rangle = \left| 0 \right\rangle](img/rangle"), ![Z\left| 1 \right\rangle = - \left| 1 \right\rangle](img/rangle"), ![X\left| + \right\rangle = \left| + \right\rangle](img/rangle"), ![X\left| - \right\rangle = - \left| - \right\rangle](img/rangle"), ![Y\left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle + i\left| 1 \right\rangle} \right) = \left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle + i\left| 1 \right\rangle} \right)](img/right)\left( {\left| 0 \right\rangle + i\left| 1 \right\rangle} \right) = \left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle + i\left| 1 \right\rangle} \right)"), 和 ![Y\left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle - i\left| 1 \right\rangle} \right) = - \left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle - i\left| 1 \right\rangle} \right)](img/right)\left( {\left| 0 \right\rangle - i\left| 1 \right\rangle} \right) = - \left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle - i\left| 1 \right\rangle} \right)").

由于对于任何 ![\left| \psi \right\rangle](img/rangle")，也成立 ![I\left| \psi \right\rangle = \left| \psi \right\rangle](img/rangle")，结果随之得出。

**(7.4)** ![Z \otimes I \otimes X](img/otimes X") 的一个可能的正交归一化特征向量基由 ![\left| 0 \right\rangle\left| 0 \right\rangle\left| + \right\rangle](img/rangle"), ![\left| 0 \right\rangle\left| 1 \right\rangle\left| - \right\rangle](img/rangle"), ![\left| 1 \right\rangle\left| 0 \right\rangle\left| - \right\rangle](img/rangle"), ![\left| 1 \right\rangle\left| 1 \right\rangle\left| + \right\rangle](img/rangle"), ![\left| 0 \right\rangle\left| 0 \right\rangle\left| - \right\rangle](img/rangle"), ![\left| 0 \right\rangle\left| 1 \right\rangle\left| + \right\rangle](img/rangle"), ![\left| 1 \right\rangle\left| 0 \right\rangle\left| + \right\rangle](img/rangle") 和 ![\left| 1 \right\rangle\left| 1 \right\rangle\left| - \right\rangle](img/rangle") 组成。前四个特征向量与特征值 ![1](img/file13.png "1") 相关联，其余的与特征值 ![- 1](img/file312.png "- 1") 相关联。

为了简化，我们可以表示 ![\( \left| i \right\rangle = \left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle + i\left| 1 \right\rangle} \right) \)](img/right)\left( {\left| 0 \right\rangle + i\left| 1 \right\rangle} \right)") 和 ![\( \left| {- i} \right\rangle = \left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle - i\left| 1 \right\rangle} \right) \)](img/right)\left( {\left| 0 \right\rangle - i\left| 1 \right\rangle} \right)"). 然后，![\( I \otimes Y \otimes Y \)](img/otimes Y") 的一个可能的标准正交基为 ![\( \left| 0 \right\rangle\left| i \right\rangle\left| i \right\rangle \)](img/rangle"), ![\( \left| 0 \right\rangle\left| {- i} \right\rangle\left| {- i} \right\rangle \)](img/rangle"), ![\( \left| 1 \right\rangle\left| i \right\rangle\left| {- i} \right\rangle \)](img/rangle"), ![\( \left| 1 \right\rangle\left| {- i} \right\rangle\left| i \right\rangle \)](img/rangle"), ![\( \left| 0 \right\rangle\left| i \right\rangle\left| {- i} \right\rangle \)](img/rangle"), ![\( \left| 0 \right\rangle\left| {- i} \right\rangle\left| i \right\rangle \)](img/rangle"), ![\( \left| 1 \right\rangle\left| i \right\rangle\left| i \right\rangle \)](img/rangle") 和 ![\( \left| 1 \right\rangle\left| {- i} \right\rangle\left| {- i} \right\rangle \)](img/rangle"). 前四个特征向量与特征值 ![\( 1 \)](img/file13.png "1") 相关联，其余的与特征值 ![\( - 1 \)](img/file312.png "- 1") 相关联。

**(7.5)** 成立 ![H\left| 0 \right\rangle = \left| + \right\rangle](img/rangle") 和 ![H\left| 1 \right\rangle = \left| - \right\rangle](img/rangle")。这证明了 ![H](img/file10.png "H") 将计算基映射到 ![X](img/file9.png "X") 的特征向量。此外，![SH\left| 0 \right\rangle = \left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle + i\left| 1 \right\rangle} \right)](img/right)\left( {\left| 0 \right\rangle + i\left| 1 \right\rangle} \right)") 和 ![SH\left| 1 \right\rangle = \left( 1\slash\sqrt{2} \right)\left( {\left| 0 \right\rangle - i\left| 1 \right\rangle} \right)](img/right)\left( {\left| 0 \right\rangle - i\left| 1 \right\rangle} \right)"), 因此 ![SH](img/file1023.png "SH") 将计算基映射到 ![Y](img/file11.png "Y") 的特征向量。

**(7.6)** 这直接源于以下事实：如果 ![{\{\left| u_{j} \right\rangle\}}_{j}](img/}}_{j}") 和 ![{\{\left| v_{k} \right\rangle\}}_{k}](img/}}_{k}") 分别是 ![A_{1}](img/file1024.png "A_{1}") 和 ![A_{2}](img/file1025.png "A_{2}") 的特征向量基，那么 ![{\{\left| u_{j} \right\rangle \otimes \left| v_{k} \right\rangle\}}_{j,k}](img/}}_{j,k}") 是 ![A_{1} \otimes A_{2}](img/otimes A_{2}") 的特征向量基。

**(7.7)** 我们问题的哈密顿量为 ![H = Z_{0}Z_{1} + Z_{1}Z_{2} + Z_{2}Z_{3} + Z_{3}Z_{4} + Z_{4}Z_{0}.](img/file1824.png "H = Z_{0}Z_{1} + Z_{1}Z_{2} + Z_{2}Z_{3} + Z_{3}Z_{4} + Z_{4}Z_{0}.") 然后，我们可以使用以下代码通过 VQE 来求解它：

```py

from qiskit.circuit.library import EfficientSU2 

from qiskit.algorithms import VQE 

from qiskit import Aer 

from qiskit.utils import QuantumInstance 

import numpy as np 

from qiskit.algorithms.optimizers import COBYLA 

from qiskit.opflow import Z, I 

seed = 1234 

np.random.seed(seed) 

H= (Z^Z^I^I^I) + (I^Z^Z^I^I) + (I^I^Z^Z^I) + (I^I^I^Z^Z) + (Z^I^I^I^Z) 

ansatz = EfficientSU2(num_qubits=5, reps=1, entanglement="linear", 

    insert_barriers = True) 

optimizer = COBYLA() 

initial_point = np.random.random(ansatz.num_parameters) 

quantum_instance = QuantumInstance(backend = 

    Aer.get_backend(’aer_simulator_statevector’)) 

vqe = VQE(ansatz=ansatz, optimizer=optimizer, 

    initial_point=initial_point, 

    quantum_instance=quantum_instance) 

result = vqe.compute_minimum_eigenvalue(H) 

print(result)

```

**(7.8)** 我们可以使用以下代码：

```py

from qiskit_nature.drivers import Molecule 

from qiskit_nature.drivers.second_quantization import \ 

    ElectronicStructureMoleculeDriver, ElectronicStructureDriverType 

from qiskit_nature.problems.second_quantization import \ 

    ElectronicStructureProblem 

from qiskit_nature.converters.second_quantization import QubitConverter 

from qiskit_nature.mappers.second_quantization import JordanWignerMapper 

from qiskit_nature.algorithms import VQEUCCFactory 

from qiskit import Aer 

from qiskit.utils import QuantumInstance 

from qiskit_nature.algorithms import GroundStateEigensolver 

import matplotlib.pyplot as plt 

import numpy as np 

quantum_instance = QuantumInstance( 

    backend = Aer.get_backend(’aer_simulator_statevector’)) 

vqeuccf = VQEUCCFactory(quantum_instance = quantum_instance) 

qconverter = QubitConverter(JordanWignerMapper()) 

solver = GroundStateEigensolver(qconverter, vqeuccf) 

energies = [] 

distances = np.arange(0.2, 2.01, 0.01) 

for d in distances: 

  mol = Molecule(geometry=[[’H’, [0., 0., -d/2]], 

                              [’H’, [0., 0., d/2]]]) 

  driver = ElectronicStructureMoleculeDriver(mol, basis=’sto3g’, 

        driver_type=ElectronicStructureDriverType.PYSCF) 

  problem = ElectronicStructureProblem(driver) 

  result = solver.solve(problem) 

  energies.append(result.total_energies) 

plt.plot(distances, energies) 

plt.title(’Dissociation profile’) 

plt.xlabel(’Distance’) 

plt.ylabel(’Energy’);

```

**(7.9)** 我们可以使用以下代码：

```py

from qiskit import * 

from qiskit.providers.aer import AerSimulator 

from qiskit.utils.mitigation import CompleteMeasFitter 

from qiskit.utils import QuantumInstance 

provider = IBMQ.load_account() 

backend = AerSimulator.from_backend( 

    provider.get_backend(’ibmq_manila’)) 

shots = 1024 

qc = QuantumCircuit(2,2) 

qc.h(0) 

qc.cx(0,1) 

qc.measure(range(2),range(2)) 

result = execute(qc, backend, shots = shots) 

print("Result of noisy simulation:") 

print(result.result().get_counts()) 

quantum_instance = QuantumInstance( 

    backend = backend, shots = shots, 

    measurement_error_mitigation_cls=CompleteMeasFitter) 

result = quantum_instance.execute(qc) 

print("Result of noisy simulation with error mitigation:") 

print(result.get_counts())

```

运行这些指令的结果如下：

```py

Result of noisy simulation: 

{’01’: 88, ’10’: 50, ’00’: 453, ’11’: 433} 

Result of noisy simulation with error mitigation: 

{’00’: 475, ’01’: 12, ’10’: 14, ’11’: 523}

```

我们知道，运行此电路的理想结果不应产生任何 ![01](img/file159.png "01") 或 ![10](img/file161.png "10") 测量。这些在噪声模拟中相当突出，但当我们使用读出错误缓解时，则不那么明显。

**(7.10)** 我们可以使用以下代码：

```py

from qiskit.opflow import Z 

from qiskit.providers.aer import AerSimulator 

from qiskit.algorithms import QAOA 

from qiskit.utils import QuantumInstance 

from qiskit import Aer, IBMQ 

from qiskit.algorithms.optimizers import COBYLA 

from qiskit.utils.mitigation import CompleteMeasFitter 

H1 = Z^Z 

provider = IBMQ.load_account() 

backend = AerSimulator.from_backend( 

    provider.get_backend(’ibmq_manila’)) 

quantum_instance = QuantumInstance(backend=backend, 

                   shots = 1024) 

qaoa = QAOA(optimizer = COBYLA(), quantum_instance=quantum_instance) 

result = qaoa.compute_minimum_eigenvalue(H1) 

print("Result of noisy simulation:",result.optimal_value) 

quantum_instance = QuantumInstance(backend=backend, 

    measurement_error_mitigation_cls=CompleteMeasFitter, 

    shots = 1024) 

qaoa = QAOA(optimizer = COBYLA(), quantum_instance=quantum_instance) 

result = qaoa.compute_minimum_eigenvalue(H1) 

print("Result of noisy simulation with error mitigation:", 

    result.optimal_value)

```

我们运行它时得到的结果如下：

```py

Result of noisy simulation: -0.8066406250000001 

Result of noisy simulation with error mitigation: -0.93359375

```

我们知道，我们哈密顿量的实际最优值是 ![- 1](img/file312.png "- 1")。因此，我们观察到，在这种情况下，噪声对 QAOA 的性能有负面影响，并且通过使用读出错误缓解可以减少这种影响。

# 第八章，什么是量子机器学习？

**(8.1)** 我们将通过反证法进行证明。我们假设存在一些系数 ![w_{1},w_{2},b](img/file1825.png "w_{1},w_{2},b") 使得

| ![0w_{1} + 1w_{2} + b = 1,\qquad 1w_{1} + 0w_{2} + b = 1,](img/qquad 1w_{1} + 0w_{2} + b = 1,") |
| --- |
| ![0w_{1} + 0w_{2} + b = 0,\qquad 1w_{1} + 1w_{2} + b = 0.](img/qquad 1w_{1} + 1w_{2} + b = 0.") |

简化后，这些等式相当于

| ![w_{2} + b = 1,\qquad w_{1} + b = 1,\qquad b = 0,\qquad w_{1} + w_{2} + b = 0.](img/qquad w_{1} + w_{2} + b = 0.") |
| --- |

前三个恒等式意味着 ![b = 0](img/file1496.png "b = 0") 和 ![w_{1} = w_{2} = 1](img/file1829.png "w_{1} = w_{2} = 1")，因此最后一个恒等式无法满足。

**(8.2)** 直方图通常是多功能且强大的选项。然而，在这种情况下，由于我们的数据集有两个特征，我们也可以使用 `plt``.``scatter` 函数绘制散点图。

**(8.3)** 函数显然是严格递增的，因为其导数是

| ![ \frac{e^{x}}{{(e^{x} + 1)}^{2}} > 0.](img/frac{e^{x}}{{(e^{x} + 1)}^{2}} > 0.") |
| --- |

此外，显然有 ![\underset{x\rightarrow\infty}{\lim}S(x) = 1](img/lim}S(x) = 1") 和 ![\underset{x\rightarrow - \infty}{\lim}S(x) = 0](img/lim}S(x) = 0").

ELU 函数是光滑的，因为 ![x](img/file269.png "x") 在 ![0](img/file12.png "0") 处的导数是 ![1](img/file13.png "1")，同样 ![e^{x} - 1](img/file1833.png "e^{x} - 1") 在 ![0](img/file12.png "0") 处的导数也是 ![1](img/file13.png "1")。这两个函数都是严格递增的，且 ![x\rightarrow\infty](img/infty") 和 ![e^{x} - 1\rightarrow - 1](img/lim}e^{x} - 1 = - 1")。

ReLU 函数的图像显然是 ![lbrack 0,\infty)](img/infty)"). 它不光滑，因为 ![ {(x)}^{\prime} = 1](img/file1837.png "{(x)}^{\prime} = 1") 而 ![ {(0)}^{\prime} = 0](img/file1838.png "{(0)}^{\prime} = 0").

**(8.4)** 在不失一般性的情况下，我们将假设 ![y = 1](img/file769.png "y = 1")（![y = 0](img/file1408.png "y = 0") 的情况完全类似）。如果 ![M_{\theta}(x) = y = 1](img/theta}(x) = y = 1")，那么 ![H(\theta;x,y) = - 1{\,\log}(1) + 0 = 0](img/theta;x,y) = - 1{\,\log}(1) + 0 = 0") 因为对于任何 ![x](img/file269.png "x") 的值，![1 - y](img/file1841.png "1 - y") 都是 ![0](img/file12.png "0")。另一方面，当 ![M_{\theta}(x)\rightarrow 0](img/theta}(x)\rightarrow 0 \right.") 时，则 ![ - {\log}(M_{\theta}(x))\rightarrow\infty](img/theta}(x))\rightarrow\infty \right.")，因此 ![H](img/file10.png "H") 也发散。

**(8.5)** 我们可以使用以下代码片段绘制损失：

```py

val_loss = history.history["val_loss"] 

train_loss = history.history["loss"] 

epochs = range(len(train_loss)) 

plt.plot(epochs, train_loss, label = "Training loss") 

plt.plot(epochs, val_loss, label = "Validation loss") 

plt.legend() 

plt.show()

```

**(8.6)** 当我们在不增加 epoch 数量的情况下降低学习率时，结果会不准确，因为算法无法采取足够的步骤达到最小值。当我们把训练数据集减少到 ![20](img/file588.png "20") 时，我们也会得到更差的结果，因为我们有过拟合。这可以通过观察验证损失的演变并注意到它在训练损失急剧下降的同时急剧上升来识别。

# 第九章，量子支持向量机

**(9.1)** 我们将证明由 ![\overset{\rightarrow}{w} \cdot \overset{\rightarrow}{x} + b = 1](img/rightarrow}{x} + b = 1") 或 ![\overset{\rightarrow}{w} \cdot \overset{\rightarrow}{x} + b = - 1](img/rightarrow}{x} + b = - 1") 特征的超平面 ![H_{1}](img/file544.png "H_{1}") 和由 ![\overset{\rightarrow}{w} \cdot \overset{\rightarrow}{x} + b = 0](img/rightarrow}{x} + b = 0") 给出的 ![H_{0}](img/file545.png "H_{0}") 之间的距离是 ![\left. 1\slash\left\| w \right\| \right.](img/right.")。结果将随后从 ![\overset{\rightarrow}{w} \cdot \overset{\rightarrow}{x} + b = \pm 1](img/pm 1") 是彼此在 ![H_{0}](img/file545.png "H_{0}") 上的投影这一事实得出。

让我们考虑一个点 ![{\overset{\rightarrow}{x}}_{0} \in H_{0}](img/in H_{0}"). ![H_{0}](img/file545.png "H_{0}") 和 ![H_{1}](img/file544.png "H_{1}") 之间的距离将是唯一一个垂直于 ![H_{0}](img/file545.png "H_{0}") 方向且连接 ![{\overset{\rightarrow}{x}}_{0}](img/rightarrow}{x}}_{0}") 到 ![H_{1}](img/file544.png "H_{1}") 中一点的向量的长度。更重要的是，由于 ![\overset{\rightarrow}{w}](img/rightarrow}{w}") 垂直于 ![H_{0}](img/file545.png "H_{0}"),这样一个向量需要是某个标量 ![\alpha](img/alpha") 的 ![\alpha\overset{\rightarrow}{w}](img/rightarrow}{w}") 的形式。让我们找到这个标量。

我们知道 ![{\overset{\rightarrow}{x}}_{0} + \alpha{\overset{\rightarrow}{x}}_{1} \in H_{1}](img/in H_{1}"), 因此我们必须有

| ![\overset{\rightarrow}{w} \cdot \left( {{\overset{\rightarrow}{x}}_{0} + \alpha\overset{\rightarrow}{w}} \right) + b = 1.](img/right) + b = 1.") |
| --- |

但考虑到![x_{0} \in H_{0}](img/in H_{0}")以及因此![w⃗ · x⃗_{0} + b = 0](img/file1851.png "w⃗ · x⃗_{0} + b = 0")，这可以进一步简化为

| ![<w⃗ · αw⃗ = 1 ⇔ α = 1/( | w | ²).](img/( | w | ²).") |
| --- | --- | --- | --- | --- |

向量![αw⃗](img/file1847.png "αw⃗")的长度将是![|α| ⋅ ||w⃗||](img/file1853.png "|α| ⋅ ||w⃗||"), 这就是![1/||w⃗||](img/||w⃗||"), 正如我们想要证明的那样。

**(9.2)** 假设核函数定义为![k(a,b) = |\langle \varphi(a) | \varphi(b) \rangle|²](img/file1854.png "k(a,b) = |\langle \varphi(a) | \varphi(b) \rangle|²")。在![C^{n}](img/file1526.png "C^{n}")中的内积是共轭对称的，因此我们必须有

| ![k(b,a) = | \langle \varphi(b) | \varphi(a) \rangle | ² = | \overline{\langle \varphi(a) | \varphi(b) \rangle} | ² = | \langle \varphi(a) | \varphi(b) \rangle | ² = k(a,b).](img/file1855.png "k(b,a) = | \langle \varphi(b) | \varphi(a) \rangle | ² = | \overline{\langle \varphi(a) | \varphi(b) \rangle} | ² = | \langle \varphi(a) | \varphi(b) \rangle | ² = k(a,b).") |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

**(9.3)** 量子态需要归一化，因此它们与自身的标量积必须为 1。

**(9.4)** 以下代码片段将实现该函数：

```py

from qiskit import * 

from qiskit . circuit import ParameterVector 

def AngleEncodingX(n): 

    x = ParameterVector("x", length = n) 

    qc = QuantumCircuit(n) 

    for i in range(n): 

        qc.rx(parameter[i], i) 

    return qc

```

# 第十章，量子神经网络

**(10.1)** 只需记住，虽然基矢量由列矩阵表示，而共轭基矢量由行矩阵表示。这样，

| ![ | 0⟩⟨0 | = \begin{pmatrix} 1 \\ 0 \\ \end{pmatrix}\begin{pmatrix} 1 & 0 \\ \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \\ \end{pmatrix}.](img/end{pmatrix}.") |
| --- | --- | --- | --- | --- |

类似地，

| ![\left | 1 \right\rangle\left\langle 1 \right | = \begin{pmatrix} 0 \\ 1 \\ \end{pmatrix}\begin{pmatrix} 0 & 1 \\ \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \\ \end{pmatrix}. |
| --- | --- | --- | --- | --- |

结果由此直接得出。

**(10.2)** 该过程将完全类似。唯一的区别在于网络的定义、设备和权重字典，可能如下所示：

```py

nqubits = 5 

dev = qml.device("default.qubit", wires=nqubits) 

def qnn_circuit(inputs, theta): 

    qml.AmplitudeEmbedding(features = [a for a in inputs], 

        wires = range(nqubits), normalize = True, pad_with = 0.) 

    TwoLocal(nqubits = nqubits, theta = theta, reps = 2) 

    return qml.expval(qml.Hermitian(M, wires = [0])) 

qnn = qml.QNode(qnn_circuit, dev, interface="tf") 

weights = {"theta": 15}

```

此外，请记住，你应该在原始数据集（`x_tr`和`y_tr`）上训练此模型，而不是在减少的数据集上！

**(10.3)** 在量子硬件上，对于大多数返回类型，可以使用有限差分法和参数平移规则。在模拟器中——在特定条件下——除了反向传播和伴随微分之外，还可以使用这些方法。

# 第十一章，两全其美：混合架构

**(11.1)** 为了包含额外的经典层，我们需要执行相同的代码，但以以下方式定义模型：

```py

model = tf.keras.models.Sequential([ 

    tf.keras.layers.Input(20), 

    tf.keras.layers.Dense(16, activation = "elu"), 

    tf.keras.layers.Dense(8, activation = "elu"), 

    tf.keras.layers.Dense(4, activation = "sigmoid"), 

    qml.qnn.KerasLayer(qnn, weights, output_dim=1) 

])

```

训练后，此模型具有相似的性能。添加经典层并没有带来非常显著的区别。

**(11.2)** 为了优化学习率和批量大小，我们可以将目标函数定义为以下形式：

```py

def objective(trial): 

    # Define the learning rate as an optimizable parameter. 

    lrate = trial.suggest_float("learning_rate", 0.001, 0.1) 

    bsize = trial.suggest_int("batch_size", 5, 50) 

    # Define the optimizer with the learning rate. 

    opt = tf.keras.optimizers.Adam(learning_rate = lrate) 

    # Prepare and compile the model. 

    model = tf.keras.models.Sequential([ 

        tf.keras.layers.Input(20), 

        tf.keras.layers.Dense(4, activation = "sigmoid"), 

        qml.qnn.KerasLayer(qnn, weights, output_dim=1) 

    ]) 

    model.compile(opt, loss=tf.keras.losses.BinaryCrossentropy()) 

    # Train it! 

    history = model.fit(x_tr, y_tr, epochs = 50, shuffle = True, 

        validation_data = (x_val, y_val), 

        batch_size = bsize, 

        callbacks = [earlystop], 

        verbose = 0 # We want TensorFlow to be quiet. 

    ) 

    # Return the validation accuracy. 

    return accuracy_score(model.predict(x_val) >= 0.5, y_val)

```

**(11.3)** 我们可以尝试使用 Optuna 找到 ![f(x) = {(x - 3)}^{2}](img/file1405.png "f(x) = {(x - 3)}^{2}") 的数值近似，如下所示：

```py

import optuna 

from optuna.samplers import TPESampler 

seed = 1234 

def objective(trial): 

    x = trial.suggest_float("x", -10, 10) 

    return (x-3)**2 

study = optuna.create_study(direction=’minimize’, 

sampler=TPESampler(seed = seed)) 

study.optimize(objective, n_trials=100)

```

当然，Optuna 并非针对（通用）函数最小化而设计，而是仅作为一个超参数优化器而构思。

**(11.4)** 设![y = 0,1](img/file1858.png "y = 0,1")为预期的标签。只需注意，![y](img/file270.png "y")在一维热编码形式中是![(1 - y,y)")，并且模型分配给![x](img/file269.png "x")属于![y](img/file270.png "y")类别的概率是![N_{\theta}{(x)}_{j}](img/theta}{(x)}_{j}"). 因此

![H(\theta;x,(1 - y,y)) = \sum\limits_{j} - y_{j}{\log}(N_{\theta}(x)_{j}) =](img/theta;x,(1 - y,y)) = \sum\limits_{j} - y_{j}{\log}(N_{\theta}(x)_{j}) =")

![- (1 - y){\,\log}(N_{\theta}(x)_{0}) - y{\,\log}(N_{\theta}(x)_{1}) =](img/file1862.png "- (1 - y){\,\log}(N_{\theta}(x)_{0}) - y{\,\log}(N_{\theta}(x)_{1}) =")

![- (1 - y){\,\log}(1 - N_{\theta}(x)_{1}) - y{\,\log}(N_{\theta}(x)_{1}),](img/file1863.png "- (1 - y){\,\log}(1 - N_{\theta}(x)_{1}) - y{\,\log}(N_{\theta}(x)_{1}),")

其中我们假设 ![N_{\theta}(x)](img/theta}(x)") 是归一化的，因此 ![N_{\theta}{(x)}_{0} + N_{\theta}{(x)}_{1} = 1](img/theta}{(x)}_{0} + N_{\theta}{(x)}_{1} = 1")。结果现在可以从以下事实得出，即在二元交叉熵中，我们考虑的概率是分配标签 ![1](img/file13.png "1") 的概率，即 ![N_{(}\theta)_{1}](img/theta)_{1}")。

**(11.5)** 在准备数据集时，我们只需使用`y`目标而不是`y_hot`目标，然后在编译模型时调用声明中给出的稀疏交叉熵损失。

**(11.6)** 您可以使用以下指令创建一个包含 1000 个样本和 20 个特征的合适数据集：

```py

x, y = make_regression(n_samples = 1000, n_features = 20)

```

然后，您可以按照以下方式构建模型：

```py

nqubits = 4 

dev = qml.device("lightning.qubit", wires = nqubits) 

@qml.qnode(dev, interface="tf", diff_method = "adjoint") 

def qnn(inputs, theta): 

    qml.AngleEmbedding(inputs, range(nqubits)) 

    TwoLocal(nqubits, theta, reps = 2) 

    return [qml.expval(qml.Hermitian(M, wires = [0]))] 

weights = {"theta": 12} 

model = tf.keras.models.Sequential([ 

    tf.keras.layers.Input(20), 

    tf.keras.layers.Dense(16, activation = "elu"), 

    tf.keras.layers.Dense(8, activation = "elu"), 

    tf.keras.layers.Dense(4, activation = "sigmoid"), 

    qml.qnn.KerasLayer(qnn, weights, output_dim=1), 

    tf.keras.layers.Dense(1) 

])

```

然后，它将使用均方误差损失函数进行训练，您可以通过`tf``.``keras``.``losses``.``MeanSquaredError` `()`访问该函数。

# 第十二章，量子生成对抗网络

**(12.1)** (1) QSVMs，QNNs 或混合 QNNs (2) QGANs。 (3) QSVMs，QNNs 或混合 QNNs。 (4) QSVMs，QNNs 或混合 QNNs。 (5) QGANs。

**(12.2)** 生成解的步骤与我们正文中所做的是类似的；只需更改定义状态 ![左| \psi_{1} \right\rangle](img/rangle") 的角度值，可能还需要增加训练周期数。

**(12.3)** 让我们考虑布洛赫球体上的一个点 ![(x,y,z)](img/file1866.png "(x,y,z)")。它的球坐标是 ![(\theta,\varphi)](img/varphi)")，使得

| ![左( x,y,z) = (\sin\theta\cos\varphi,\sin\theta\sin\varphi,\cos\theta) \right.](img/file1868.png "左( x,y,z) = (\sin\theta\cos\varphi,\sin\theta\sin\varphi,\cos\theta) \right.") |
| --- |

并且具有那些球面布洛赫球坐标的状态是处于状态 ![左. {|\psi\rangle} = {\cos}(\theta/2){|0\rangle} + e^{i\phi}{\sin}(\theta/2){|1\rangle}. \right.](img/2){|0\rangle} + e^{i\phi}{\sin}(\theta/2){|1\rangle}. \right.") 在该状态下 ![Z = \left| 0 \right\rangle\left\langle 0 \right| - \left| 1 \right\rangle\left\langle 1 \right|](img/right|") 的期望值是

| ![左. \left\langle \psi \right | Z\left | \psi \right\rangle = {\cos}^{2}(\theta\slash 2) - e^{- i\varphi + i\varphi}{\sin}^{2}(\theta\slash 2) = \cos\theta = z. \right.](img/slash 2) - e^{- i\varphi + i\varphi}{\sin}^{2}(\theta\slash 2) = \cos\theta = z. \right.") |
| --- | --- | --- | --- | --- |

关于期望值 ![Y = i\left| 1 \right\rangle\left\langle 0 \right| - i\left| 0 \right\rangle\left\langle 1 \right|](img/right|"), 我们有 ![\left. {\langle\psi|}Y{|\psi\rangle} = ie^{- i\phi}{\sin}(\theta/2){\,\cos}(\theta/2) - ie^{i\phi}{\sin}(\theta/2){\,\cos}(\theta/2) \right.](img/2){\,\cos}(\theta/2) - ie^{i\phi}{\sin}(\theta/2){\,\cos}(\theta/2) \right.")

![\left. = i(e^{- i\phi} - e^{i\phi})({\sin}(\theta/2){\,\cos}(\theta/2)) \right.](img/phi})({\sin}(\theta/2){\,\cos}(\theta/2)) \right.")

![\left. = {\sin\,}\phi \cdot 2{\,\sin}(\theta/2){\,\cos}(\theta/2) = {\sin\,}\phi{\,\sin\,}\theta = y. \right.](img/2){\,\cos}(\theta/2) = {\sin\,}\phi{\,\sin\,}\theta = y. \right.")

最后，关于期望值 ![X = \left| 1 \right\rangle\left\langle 0 \right| + \left| 0 \right\rangle\left\langle 1 \right|](img/right|")，

![\left. {\langle\psi|}X{|\psi\rangle} = e^{- i\phi}{\sin}(\theta/2){\,\cos}(\theta/2) + e^{i\phi}{\sin}(\theta/2){\,\cos}(\theta/2) \right.](img/2){\,\cos}(\theta/2) + e^{i\phi}{\sin}(\theta/2){\,\cos}(\theta/2) \right.")

![\left. = (e^{- i\phi} + e^{i\phi})({\sin}(\theta/2){\,\cos}(\theta/2)) \right.](img/phi})({\sin}(\theta/2){\,\cos}(\theta/2)) \right.")

![\left. = {\cos\,}\phi \cdot 2{\,\sin}(\theta/2){\,\cos}(\theta/2) = {\cos\,}\phi{\,\sin\,}\theta = x. \right.](img/2){\,\cos}(\theta/2) = {\cos\,}\phi{\,\sin\,}\theta = x. \right.")
