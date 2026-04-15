from MMD import get_error_MMD, ED_k
import explicitConstrainedFlow as m
import oneshotFlow as oneshot
import implicitConstrainedFlow as imp


def expe(X_0=None, **kwargs):
    # Determine which dynamic function to use based on the provided dynamic model
    if kwargs['dynamic'].__name__ in ['explicitDynamic']: 
        map = m.buildExplicitDynamic(X_0=X_0, **kwargs)
    elif kwargs['dynamic'].__name__ in ['oneshotDynamic']: 
        map = oneshot.buildOneshotDynamic(X_0=X_0, **kwargs)
    elif kwargs['dynamic'].__name__ in ['implicitDyanmic']:
        map = imp.implicitDyanmic(X_0=X_0, **kwargs)
    else:
        raise NotImplementedError

    if kwargs['generator_target'] is not None:
        if 'target' in kwargs and kwargs['target'] is not None:
            gamma = kwargs['target'].detach().cpu().numpy()
        else:
            gamma = kwargs['generator_target'].next().detach().cpu().numpy()

        if kwargs['n_batch_big_MMD'] is not None:
            gen_rho0 = kwargs['generator']
            gen_rho0.n_batch = kwargs['n_batch_big_MMD']
            gen_gamma = kwargs['generator_target']
            gen_gamma.n_batch = kwargs['n_batch_big_MMD']
            rho0 = gen_rho0.next()
            gamma = gen_gamma.next()

            if type(rho0) is tuple:
                rho0 = rho0[0]
            rho_infty = map.forward(rho0)
            mmd_big = get_error_MMD(rho_infty.detach().cpu().numpy(), gamma.detach().cpu().numpy(), ED_k())
    return mmd_big
