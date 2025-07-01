import torch

def simple_gradients():
    """
    Простые вычисления с градиентами для задания 2.1

    Returns:
        tuple: Кортеж с результатами:
            - x, y, z: исходные тензоры
            - f: значение функции
            - grad_x, grad_y, grad_z: градиенты
            - analitic_grads: аналитические градиенты
    """
    # Создайте тензоры x, y, z с requires_grad=True
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    # Вычисляем функцию f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z

    # Находим градиенты по всем переменным
    f.backward()

    # Получаем градиенты
    grad_x = x.grad
    grad_y = y.grad
    grad_z = z.grad

    # Проверяем результат аналитически
    # df/dx = 2x + 2yz
    # df/dy = 2y + 2xz
    # df/dz = 2z + 2xy
    analitic_grad_x = 2 * x.item() + 2 * y.item() * z.item()
    analitic_grad_y = 2 * y.item() + 2 * x.item() * z.item()
    analitic_grad_z = 2 * z.item() + 2 * x.item() * y.item()
    analitic_grads = (analitic_grad_x, analitic_grad_y, analitic_grad_z)

    return x, y, z, f, (grad_x, grad_y, grad_z), analitic_grads


def mse_loss_gradients():
    """
    Градиент функции потерь для задания 2.2

    Returns:
        tuple: Кортеж с результатами:
            - w, b: параметры модели
            - x, y_true: входные данные и целевые значения
            - y_pred: предсказания модели
            - loss: значение функции потерь
            - grad_w, grad_b: градиенты по параметрам
    """
    # Инициализируем параметры с requires_grad=True
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)

    # Создаем входные данные и целевые значения
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])  # Идеальная зависимость y = 2x

    # Вычисляем предсказания: y_pred = w * x + b
    y_pred = w * x + b

    # Вычисляем MSE loss
    # MSE = (1/n) * Σ(y_pred - y_true)^2
    loss = torch.mean((y_pred - y_true) ** 2)

    # Вычисляем градиенты
    loss.backward()

    # Получаем градиенты
    grad_w = w.grad
    grad_b = b.grad

    return w, b, x, y_true, y_pred, loss, (grad_w, grad_b)


def chain_rule_example():
    """
    Применение цепного правила (задание 2.3)

    Returns:
        tuple: Кортеж с результатами:
            - x: входной тензор
            - f: значение функции
            - grad: градиент вычисленный через backward()
            - grad_check: градиент проверенный через torch.autograd.grad
            - analytic_grad: аналитический градиент
    """
    # Создаем тензор с requires_grad=True
    x = torch.tensor(2.0, requires_grad=True)

    # Вычисляем составную функцию f(x) = sin(x^2 + 1)
    f = torch.sin(x ** 2 + 1)

    # Вычисляем градиент через backward()
    f.backward(retain_graph=True)  # Добавляем retain_graph=True
    grad = x.grad

    # Проверяем градиент с помощью torch.autograd.grad
    # Нужно создать новый граф вычислений
    x2 = torch.tensor(2.0, requires_grad=True)
    f2 = torch.sin(x2 ** 2 + 1)
    grad_check = torch.autograd.grad(f2, x2)[0]

    # Аналитическая проверка
    # df/dx = cos(x^2 + 1) * 2x
    analytic_grad = torch.cos(x ** 2 + 1) * 2 * x

    return x, f, grad, grad_check, analytic_grad


def test_functions():
    """Тестирование всех функций"""
    print("=== Тестирование простых вычислений с градиентами ===")
    x, y, z, f, grads, analitic_grads = simple_gradients()
    print("\n1.1 Исходные значения:")
    print(f"x = {x.item()}, y = {y.item()}, z = {z.item()}")
    print("\n1.2 Значение функции f:", f.item())
    print("\n1.3 Градиенты (autograd):")
    print(f"df/dx = {grads[0].item()}, df/dy = {grads[1].item()}, df/dz = {grads[2].item()}")
    print("\n1.4 Аналитические градиенты:")
    print(f"df/dx = {analitic_grads[0]}, df/dy = {analitic_grads[1]}, df/dz = {analitic_grads[2]}")

    print("\n=== Тестирование градиентов MSE ===")
    w, b, x, y_true, y_pred, loss, grads = mse_loss_gradients()
    print("\n2.1 Параметры модели:")
    print(f"w = {w.item()}, b = {b.item()}")
    print("\n2.2 Входные данные (x):", x)
    print("Целевые значения (y_true):", y_true)
    print("Предсказания (y_pred):", y_pred)
    print("\n2.3 Значение MSE loss:", loss.item())
    print("\n2.4 Градиенты:")
    print(f"dL/dw = {grads[0].item()}, dL/db = {grads[1].item()}")

    print("\n=== Тестирование цепного правила ===")
    x, f, grad, grad_check, analytic_grad = chain_rule_example()
    print("\n3.1 Входное значение x:", x.item())
    print("3.2 Значение функции f(x) = sin(x^2 + 1):", f.item())
    print("\n3.3 Градиент через backward():", grad.item())
    print("3.4 Градиент через autograd.grad:", grad_check.item())
    print("3.5 Аналитический градиент:", analytic_grad.item())


if __name__ == "__main__":
    # Проверяем доступность GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Переносим все тензоры на выбранное устройство
    torch.set_default_device(device)

    # Запускаем тестирование
    test_functions()
