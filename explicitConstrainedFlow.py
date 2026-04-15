import torch
from mongeMap import ICNNMap


FUNCTIONS = {'ICNNMap': ICNNMap}


def solve_inner_optim_problem(map, t_to_learn, X_t, X_0, v_t, nb_grad_step, generator, dynamic_to_correct, mapClass, **kwargs):
    lr_inner = kwargs['lr_inner_explicit']
    #optimizer = torch.optim.SGD(map.parameters(), lr=lr_modin)
    optimizer = torch.optim.Adam(map.parameters(), lr=lr_inner)
    losses = []
    # remember map at previous state
    if mapClass == 'ICNNMap':
        map_previous = FUNCTIONS[mapClass](map.hidden_dim, map.depth, n_timesteps=map.n_timesteps, final_time=map.final_time)
    else:
        raise NotImplementedError
    map_previous.load_state_dict(map.state_dict())
    map_previous.current_time = map.current_time - 1
    # Perform optimization
    for i in range(nb_grad_step):
        X_0 = generator.next()
        if type(X_0) is tuple: X_0 = X_0[0]
        X_t = map_previous.forward(X_0,t_to_learn-1)
        v_t = dynamic_to_correct(X_t, **kwargs)

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((map.forward(X_0,t_to_learn)-(X_t+map.dt*v_t))**2)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        loss = closure()
        losses.append(loss.detach().numpy())

    return losses


def explicitDynamic(gen, dynamic_to_correct, n_timesteps,final_time,nb_grad_step, X_0=None, mapClass='ICNNMap', **kwargs):
    device = kwargs['device']
    if X_0 is None:
        X_0 = gen.next()
    else:
        assert gen.n_batch == X_0.shape[0]

    if mapClass in FUNCTIONS:
        map = FUNCTIONS[mapClass](n_timesteps=n_timesteps,final_time=final_time, **kwargs).to(device)
    else:
        raise NotImplementedError
    torch.autograd.set_detect_anomaly(True)

    for j in range(n_timesteps):
        X_t = map.forward(X_0, j, save=True).detach().clone()
        v_t = dynamic_to_correct(X_t, **kwargs)

        if mapClass in FUNCTIONS:
            map.current_time += 1
            _ = solve_inner_optim_problem(map=map, t_to_learn=j+1, X_t=X_t, X_0=X_0, v_t=v_t, nb_grad_step=nb_grad_step, generator=gen, dynamic_to_correct=dynamic_to_correct, mapClass=mapClass, **kwargs)
        
    return map


def buildExplicitDynamic(generator, dynamic, dynamic_to_correct, n_steps, final_time, n_batch, nb_grad_step, X_0, mapClass, **kwargs):
    generator.n_batch = n_batch
    
    map = dynamic(generator, dynamic_to_correct, n_steps, final_time, nb_grad_step, X_0, mapClass, **kwargs)

    return map
