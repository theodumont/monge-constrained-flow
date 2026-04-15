import torch

from mongeMap import ICNNMap
from dynamics import estimateGradKL


def oneshotDynamic(generator, dynamic_to_correct,nb_grad_step_oneshot, X_0=None, mapClass='LSEMap', **kwargs):
    device = kwargs['device']
    if X_0 is None:
        X_0 = generator.next()
    else:
        assert generator.n_batch == X_0.shape[0]

    FUNCTIONS = {'ICNNMap': ICNNMap}
    if mapClass in FUNCTIONS:
        map = FUNCTIONS[mapClass](n_timesteps=None,final_time=None, **kwargs).to(device)
    else:
        raise NotImplementedError
    target = kwargs['target']
    overfit = kwargs['overfit']
    torch.autograd.set_detect_anomaly(True)

    optimizer = torch.optim.SGD(map.parameters(), lr=kwargs['lr'])  # Change this line to get ADAM

    res = [(X_0.detach().numpy(), target.detach().numpy() if target is not None else None, X_0.detach().numpy())]

    loss_values = []
    
    for j in range(nb_grad_step_oneshot):
        optimizer.zero_grad()
        if not overfit:
            X_0 = generator.next()
            if type(X_0) is tuple:
                X_0 = X_0[0]
        T_X = map.forward(X_0)
        if dynamic_to_correct.__name__ not in ["langevinDynamic"]:
            # compute loss
            loss_value = dynamic_to_correct(T_X, target)
            # backpropagate
            loss_value.backward(retain_graph=True)
            loss_values.append(loss_value.item())
        else:
            # compute gradient manually
            grad = estimateGradKL(map, T_X, target)
            for p, g in zip(map.parameters(), grad):
                if p.grad is not None:
                    p.grad.zero_()
                p.grad = g.clone()  # assign new gradient
        optimizer.step()
        res.append((X_0.detach().numpy(), target.detach().numpy() if target is not None else None, T_X.detach().numpy()))

    return map



def buildOneshotDynamic(dynamic, **kwargs):
    map = dynamic(**kwargs)
    return map