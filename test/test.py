import torch

def test():
    print(f"Test Case 1: ")
    x = torch.Tensor([14.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()

    print(f"y.data = {y.data.item()}")
    print(f"x.grad = {x.grad.item()}")

    print(f"\nTest Case 2: ")
    a = torch.Tensor([-12.0]).double()
    b = torch.Tensor([6.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()

    print(f"g.data = {g.data.item()}")
    print(f"a.grad = {a.grad.item()}")
    print(f"b.grad = {b.grad.item()}")


if __name__ == "__main__":
    test()