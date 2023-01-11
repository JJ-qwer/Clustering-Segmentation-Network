import numpy as np
from sklearn import linear_model
from scipy.cluster.vq import vq, whiten, kmeans


def KSVD(Y, dict_size,
         max_iter=10,
         sparse_rate=0.2,
         tolerance=1e-6):
    assert (dict_size <= Y.shape[1])

    def dict_update(y, d, x):
        assert (d.shape[1] == x.shape[0])

        for i in range(x.shape[0]):
            index = np.where(np.abs(x[i, :]) > 1e-7)[0]

            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0]
            for j, k in enumerate(index):
                x[i, k] = s[0] * v[0, j]
        return d, x

    # initialize dictionary
    if dict_size > Y.shape[0]:
        dic = Y[:, np.random.choice(Y.shape[1], dict_size, replace=False)]
    else:
        u, s, v = np.linalg.svd(Y)
        dic = u[:, :dict_size]

    # print('dict shape:', dic.shape)

    n_nonzero_coefs_each_code = int(sparse_rate * dict_size) if int(sparse_rate * dict_size) > 0 else 1
    for i in range(max_iter):
        x = linear_model.orthogonal_mp(dic, Y, n_nonzero_coefs=n_nonzero_coefs_each_code)
        e = np.linalg.norm(Y - dic @ x)
        if e < tolerance:
            break
        dict_update(Y, dic, x)

    sparse_code = linear_model.orthogonal_mp(dic, Y, n_nonzero_coefs=n_nonzero_coefs_each_code)

    return dic, sparse_code


def col2dict(img, flag, w, l, change):
    [ylen, xlen] = img.shape
    d = w//2
    flag[0:d, :] = 0
    flag[ylen-d:ylen, :] = 0
    upper = np.unique(np.where(flag==1, img, 0))
    mat = np.zeros((w, l))
    n = 0
    for i in range(1, l+1):
        i_where = np.argwhere(img==upper[i])
        for x in i_where:
            if x[0] >= d and x[0] < ylen-d:
                mat[:, n] = img[x[0]-d:x[0]+d+1, x[1]]
                n += 1
            if n == l:
                break
        if n == l:
            break
    return mat/255


def patch2mat(img, uncertain, w):
    d = w//2
    x_where_ori = np.argwhere(uncertain==1)
    img = np.pad(img, ((d,d),(d,d)), mode='constant')
    uncertain = np.pad(uncertain, ((d, d), (d, d)), mode='constant')
    x_where = np.argwhere(uncertain==1)
    l = len(x_where)
    mat = np.zeros((w**2, l))
    for i in range(l):
        mat[:, i] = img[x_where[i][0]-d:x_where[i][0]+d+1, x_where[i][1]-d:x_where[i][1]+d+1].reshape((w**2), order='F')
    return mat/255, x_where_ori


def nearpatch2dict(img, flag, w, n_cluster, sub_length, change):
    d = w // 2
    # select original patch
    patches = []

    upper = np.unique(np.where(flag==1, img, 0))

    for i in range(len(upper)):
        i_where = np.argwhere(img==upper[i])
        for x in i_where:
            if np.sum(flag[x[0]-d:x[0]+d+1, x[1]-d:x[1]+d+1]) == w**2:
                patches.append(img[x[0]-d:x[0]+d+1, x[1]-d:x[1]+d+1].reshape((w**2), order='F'))
            if len(patches) == 10*n_cluster*sub_length:
                break
        if len(patches) == 10 * n_cluster * sub_length:
            break
    patches = np.array(patches)

    # cal var
    vars = []
    for i in range(len(patches)):
        vars.append(np.var(patches[i]))
    vars = np.array(vars)
    vars_mean = np.mean(vars)
    new_patches = []
    for i in range(len(patches)):
        if vars[i] > vars_mean * 0.75:
            new_patches.append(patches[i])
    patches = np.array(new_patches)

    if len(patches) < n_cluster:
        mat = col2dict(img, flag, w**2, 100, change)
        D, _ = KSVD(mat, 80)
        return D / 255

    # cluster
    patches_whiten = whiten(patches)
    center, _ = kmeans(patches_whiten, n_cluster)
    clus, _ = vq(patches_whiten, center)

    # select patch to train
    D = np.zeros((w**2, 1))
    for i in range(n_cluster):
        idx = np.argwhere(clus == i)
        matx = np.array([patches[u[0]] for u in idx]).T
        if matx.shape[1] >= sub_length:
            # matx = pca(matx, sub_length)
            matx, _ = KSVD(matx, sub_length)
            D = np.concatenate((D, matx), axis=1)
    D = D[:, 1:]
    return D / 255


def sparse_fuse(img1, img2, img_ori, dict_w, n_cluster, sub_length):
    gray = np.unique(img2)
    img2 = np.where(img2 == gray[0], 0, img2)
    img2 = np.where(img2 == gray[-1], 255, img2)
    img2 = np.where((img2 != 0) & (img2 != 255), 127, img2)

    gray = np.unique(img1)
    img1 = np.where(img1 == gray[0], 0, img1)
    img1 = np.where(img1 == gray[-1], 255, img1)
    img1 = np.where((img1 != 0) & (img1 != 255), 127, img1)

    change = np.where((img1 == 255) & (img2 == 255), 1, 0)

    unchange = np.where((img1 == 0) & (img2 == 0), 1, 0)

    uncertain = np.where((img1 == 127) & (img2 == 127), 1, 0)
    uncertain = np.where((img1 == 255) & (img2 == 0), 1, uncertain)
    uncertain = np.where((img1 == 0) & (img2 == 255), 1, uncertain)

    if np.sum(uncertain) != 0:
        D_c = nearpatch2dict(img_ori, change, dict_w, n_cluster, sub_length, 1)
        D_u = nearpatch2dict(img_ori, unchange, dict_w, n_cluster, sub_length, 0)
        mat_uncertain, x_where = patch2mat(img_ori, uncertain, dict_w)

        ###############################################################################################

        X_c = linear_model.orthogonal_mp(D_c, mat_uncertain)
        Y_c = D_c @ X_c
        l_c = np.sum(np.square(Y_c - mat_uncertain), axis=0)

        X_u = linear_model.orthogonal_mp(D_u, mat_uncertain)
        Y_u = D_u @ X_u
        l_u = np.sum(np.square(Y_u - mat_uncertain), axis=0)

        out_img = np.where(
            ((img1 == 255) & (img2 == 255)) | ((img1 == 255) & (img2 == 127)) | ((img1 == 127) & (img2 == 255)), 255, 0)
        out_img = np.where(((img1 == 0) & (img2 == 0)) | ((img1 == 0) & (img2 == 127)) | ((img1 == 127) & (img2 == 0)),
                           0, out_img)
        out_img = np.where(uncertain == 1, 127, out_img)

        for i in range(x_where.shape[0]):
            if l_c[i] >= l_u[i]:
                out_img[x_where[i][0], x_where[i][1]] = 255
            else:
                out_img[x_where[i][0], x_where[i][1]] = 0
    else:
        out_img = np.where(
            ((img1 == 255) & (img2 == 255)) | ((img1 == 255) & (img2 == 127)) | ((img1 == 127) & (img2 == 255)), 255, 0)
        out_img = np.where(((img1 == 0) & (img2 == 0)) | ((img1 == 0) & (img2 == 127)) | ((img1 == 127) & (img2 == 0)),
                           0, out_img)

    return out_img















