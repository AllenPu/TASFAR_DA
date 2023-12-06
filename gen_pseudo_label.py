import numpy as np

# Return uncertainty threshold, q parameter
def q_func(source_y, eta=0.9):
    pass

def con_classifier(target_y, thresh):
    pass

def density_map_construct():
    pass

def pseudo_label_gen():
    pass

def generator(target_y, thresh, q):
    set_c, set_u = con_classifier(target_y, thresh)
    map = density_map_construct(set_c, q)
    pseudo_y = pseudo_label_gen(map, set_u)
    return pseudo_y

# Pseudo-label generation using housing-price prediction dataset
if __name__ == "__main__":
    # ratio for uncertainty threshold
    eta = 0.9
    # 
    source_y = []
    # 
    target_y = []
    
    u_thresh, q = q_func(source_y, eta=eta)
    generator(target_y, eta=eta)