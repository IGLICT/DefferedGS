import renderutils as ru
import texture as dr
import jittor as jt
jt.flags.use_cuda=1
# input = jt.ones((6,16,16,3))*0.9
# diffuse = ru.diffuse_cubemap(input)
# print(diffuse)
reflvec = jt.ones((1, 1, 640000, 3))*0.8
in1 = jt.ones((1, 6, 64, 64, 3))*0.8
list1 = jt.ones((1, 6, 32, 32, 3))*0.8
list2 = jt.ones((1, 6, 16, 16, 3))*0.8
ln2 = jt.ones((1, 1, 640000))
spec = dr.texture(in1, reflvec.contiguous(), filter_mode='linear', boundary_mode='cube')
print(spec)
spec = dr.texture(in1, reflvec.contiguous(), mip=[list1,list2], mip_level_bias=ln2, filter_mode='linear-mipmap-linear', boundary_mode='cube')
print(spec)