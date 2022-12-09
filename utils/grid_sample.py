import torch
''' see the Section "Mesh-agnostic representations through interpolation" in the paper.
            below codes are about cosine/linear interpolation in 2d and 3d case. '''
def grid_sample_2d(input, grid, step='cosine', offset=True):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    N, C, IH, IW = input.shape
    _, H, W, _ = grid.shape

    if step=='bilinear':
        step_f = lambda x: x
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError

    ''' (iy,ix) will be the indices of the input
            1. normalize coordinates 0 to 1 (from -1 to 1)
            2. scaling to input size
            3. adding offset to make non-zero derivative interpolation '''
    ix = grid[..., 0]
    iy = grid[..., 1]
    if offset:
        offset = torch.linspace(0,1-(1/N),N).reshape(N,1,1).to('cuda')
        iy = ((iy+1)/2)*(IH-2) + offset
        ix = ((ix+1)/2)*(IW-2) + offset

    else:
        iy = ((iy+1)/2)*(IH-1)
        ix = ((ix+1)/2)*(IW-1)
    
    # compute corner indices
    with torch.no_grad():
        ix_left = torch.floor(ix)
        ix_right = ix_left + 1
        iy_top = torch.floor(iy)
        iy_bottom = iy_top + 1

    # compute weights
    dx_right = step_f(ix_right-ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom-iy)
    dy_top = 1 - dy_bottom

    nw = dx_right*dy_bottom
    ne = dx_left*dy_bottom
    sw = dx_right*dy_top
    se = dx_left*dy_top

    # sanity checking
    with torch.no_grad():
        torch.clamp(ix_left, 0, IW-1, out=ix_left)
        torch.clamp(ix_right, 0, IW-1, out=ix_right)
        torch.clamp(iy_top, 0, IH-1, out=iy_top)
        torch.clamp(iy_bottom, 0, IH-1, out=iy_bottom)

    # look up values
    input = input.view(N, C, IH*IW)
    nw_val = torch.gather(input, 2, (iy_top * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    ne_val = torch.gather(input, 2, (iy_top * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))
    sw_val = torch.gather(input, 2, (iy_bottom * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    se_val = torch.gather(input, 2, (iy_bottom * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))

    # 2d_cosine/bilinear interpolation
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val



def grid_sample_3d(input, grid, step='cosine', offset=False):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    N, C, IT, IH, IW = input.shape
    _, H, W, _ = grid.shape
    if step=='trilinear':
        step_f = lambda x: x
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError
    ''' (iy,ix) will be the indices of the input
            1. normalize coordinates 0 to 1 (from -1 to 1)
            2. scaling to input size
            3. adding offset to make non-zero derivative interpolation '''
    it = grid[..., 0]
    ix = grid[..., 1]
    iy = grid[..., 2]
   
    if offset:
        offset = torch.linspace(0,(1-(1/(N))),N).reshape(N,1,1)
        it = ((it+1)/2)*(IT-2) + offset
        ix = ((ix+1)/2)*(IW-2) + offset
        iy = ((iy+1)/2)*(IH-2) + offset
    else:
        it = ((it+1)/2)*(IT-1)
        ix = ((ix+1)/2)*(IW-1)
        iy = ((iy+1)/2)*(IH-1)

    
    with torch.no_grad():
        it_nw_front = torch.floor(it)
        ix_nw_front = torch.floor(ix)
        iy_nw_front = torch.floor(iy)
        
        it_sw_front = it_nw_front
        ix_sw_front = ix_nw_front
        iy_sw_front = iy_nw_front+1

        it_ne_front = it_nw_front
        ix_ne_front = ix_nw_front+1
        iy_ne_front = iy_nw_front

        it_se_front = it_nw_front
        ix_se_front = ix_nw_front+1
        iy_se_front = iy_nw_front+1

        it_nw_back = it_nw_front+1
        ix_nw_back = ix_nw_front
        iy_nw_back = iy_nw_front

        it_ne_back = it_nw_front+1
        ix_ne_back = ix_nw_front+1
        iy_ne_back = iy_nw_front

        it_sw_back = it_nw_front+1
        ix_sw_back = ix_nw_front
        iy_sw_back = iy_nw_front+1

        it_se_back = it_nw_front+1
        ix_se_back = ix_nw_front+1
        iy_se_back = iy_nw_front+1

    # compute 3d weights
    step_it = step_f(it_se_back - it)
    step_ix = step_f(ix_se_back - ix)
    step_iy = step_f(iy_se_back - iy)
    
    nw_front = step_it * step_ix * step_iy
    ne_front = step_it * (1-step_ix) * (step_iy)
    sw_front = step_it * (step_ix) * (1-step_iy)
    se_front = step_it * (1-step_ix) * (1-step_iy)

    nw_back = (1-step_it) * step_ix * step_iy
    ne_back = (1-step_it) * (1-step_ix) * (step_iy)
    sw_back = (1-step_it) * (step_ix) * (1-step_iy)
    se_back = (1-step_it) * (1-step_ix) * (1-step_iy)
    
    # sanity checking
    with torch.no_grad():
        torch.clamp(ix_nw_front, 0, IH-1, out=ix_nw_front)
        torch.clamp(iy_nw_front, 0, IW-1, out=iy_nw_front)
        torch.clamp(it_nw_front, 0, IT-1, out=it_nw_front)
        
        torch.clamp(ix_ne_front, 0, IH-1, out=ix_ne_front)
        torch.clamp(iy_ne_front, 0, IW-1, out=iy_ne_front)
        torch.clamp(it_ne_front, 0, IT-1, out=it_ne_front)
        
        torch.clamp(ix_sw_front, 0, IH-1, out=ix_sw_front)
        torch.clamp(iy_sw_front, 0, IW-1, out=iy_sw_front)
        torch.clamp(it_sw_front, 0, IT-1, out=it_sw_front)
        
        torch.clamp(ix_se_front, 0, IH-1, out=ix_se_front)
        torch.clamp(iy_se_front, 0, IW-1, out=iy_se_front)
        torch.clamp(it_se_front, 0, IT-1, out=it_se_front)

        torch.clamp(ix_nw_back, 0, IH-1, out=ix_nw_back)
        torch.clamp(iy_nw_back, 0, IW-1, out=iy_nw_back)
        torch.clamp(it_nw_back, 0, IT-1, out=it_nw_back)
        
        torch.clamp(ix_ne_back, 0, IH-1, out=ix_ne_back)
        torch.clamp(iy_ne_back, 0, IW-1, out=iy_ne_back)
        torch.clamp(it_ne_back, 0, IT-1, out=it_ne_back)
        
        torch.clamp(ix_sw_back, 0, IH-1, out=ix_sw_back)
        torch.clamp(iy_sw_back, 0, IW-1, out=iy_sw_back)
        torch.clamp(it_sw_back, 0, IT-1, out=it_sw_back)
        
        torch.clamp(ix_se_back, 0, IH-1, out=ix_se_back)
        torch.clamp(iy_se_back, 0, IW-1, out=iy_se_back)
        torch.clamp(it_se_back, 0, IT-1, out=it_se_back)
    
    
    input = input.view(N, C, IT*IH*IW)
    
    # correct
    nw_front_val = torch.gather(input, 2, (iy_nw_front * IH*IT + (ix_nw_front * IT + it_nw_front)).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_front_val = torch.gather(input, 2, (iy_ne_front * IH*IT + (ix_ne_front * IT + it_ne_front)).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_front_val = torch.gather(input, 2, (iy_sw_front * IH*IT + (ix_sw_front * IT + it_sw_front)).long().view(N, 1, H * W).repeat(1, C, 1))
    se_front_val = torch.gather(input, 2, (iy_se_front * IH*IT + (ix_se_front * IT + it_se_front)).long().view(N, 1, H * W).repeat(1, C, 1))
    nw_back_val = torch.gather(input, 2, (iy_nw_back * IH*IT + (ix_nw_back * IT + it_nw_back)).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_back_val = torch.gather(input, 2, (iy_ne_back * IH*IT + (ix_ne_back * IT + it_ne_back)).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_back_val = torch.gather(input, 2, (iy_sw_back * IH*IT + (ix_sw_back * IT + it_sw_back)).long().view(N, 1, H * W).repeat(1, C, 1))
    se_back_val = torch.gather(input, 2, (iy_se_back * IH*IT + (ix_se_back * IT + it_se_back)).long().view(N, 1, H * W).repeat(1, C, 1))

    
    # 3d_cosine/trilinear interpolation
    out_val = ((nw_front_val.view(N, C, H, W) * nw_front.view(N, 1, H, W)) + 
               (ne_front_val.view(N, C, H, W) * ne_front.view(N, 1, H, W)) +
               (sw_front_val.view(N, C, H, W) * sw_front.view(N, 1, H, W)) +
               (se_front_val.view(N, C, H, W) * se_front.view(N, 1, H, W)) + 
               (nw_back_val.view(N, C, H, W) * nw_back.view(N, 1, H, W)) + 
               (ne_back_val.view(N, C, H, W) * ne_back.view(N, 1, H, W)) +
               (sw_back_val.view(N, C, H, W) * sw_back.view(N, 1, H, W)) +
               (se_back_val.view(N, C, H, W) * se_back.view(N, 1, H, W))) 
    return out_val