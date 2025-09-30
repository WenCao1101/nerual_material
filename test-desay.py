import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr
import torch
import torch.nn as nn
import math
import numpy as np
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG10_INV=1.0 / math.log(10)

def mse(a, b):
    return dr.mean(dr.sqr(a - b))

def psnr(a, b):
    return -10.0 * LOG10_INV * dr.log(mse(a, b) + 1e-8)                

def to_scalar(x):
    return float(dr.detach(dr.eval(x)))

class MyDiffuseBSDF(mi.BSDF):
    """Custom Diffuse BSDF implementation copied from Mitsuba

    Note: this is unused in this project, kept for reference only
    """

    def __init__(self, props: mi.Properties):
        mi.BSDF.__init__(self, props)
        self.diffuse = mi.Color3f(props.get('diffuse', mi.Color3f(0.5)))
        self.m_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]      
    
    def sample(self, ctx, si, sample1, sample2, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0

        bs = mi.BSDFSample3f()
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.BSDFFlags.DiffuseReflection
        bs.sampled_component = 0
        value=self.diffuse * dr.inv_pi * cos_theta_i
        return (bs, dr.select(active & (bs.pdf > 0.0), value, mi.Vector3f(0)))
    def eval(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return mi.Vector3f(0)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        value = self.diffuse * dr.inv_pi * cos_theta_i

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), value, mi.Vector3f(0))
    
    def eval_pdf(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return 0.0

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)
    def pdf(self, ctx, si, wo, active):
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return 0.0

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)
    
    def traverse(self,callback):
        mi.BSDF.traverse(self, callback)
        callback.put_parameter('diffuse', self.diffuse,mi.ParamFlags.Differentiable)
    def to_string(self):
        return f"MyDiffuseBSDF[diffuse={self.diffuse}]"
mi.register_bsdf("mydiffuse", lambda props: MyDiffuseBSDF(props))

my_bsdf = MyDiffuseBSDF(mi.Properties({'diffuse': mi.Color3f(0.2, 0.9, 0.2)}))   
my_bsdf2 = MyDiffuseBSDF(mi.Properties({'diffuse': mi.Color3f(0.3, 0.4, 0.2)})) 
print(my_bsdf)
scene=mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'path'},
    'sensor':  {
        'type': 'perspective',
        'to_world': mi.Transform4f.look_at(
                        origin=(0, -5, 5),
                        target=(0, 0, 0),
                        up=(0, 0, 1)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width':  128,
            'height': 128,
        },
    },
    'sphere': {
        'type': 'sphere',
        'bsdf': 'my_bsdf'},
    'box': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.translate([0, 0, -1]).scale(1),
        'bsdf':'my_bsdf2'},    
    'light': {
        'type': 'constant',
        'radiance':0.99,
    }
})

key_to_optimizer='sphere.bsdf.diffuse'
optimizer_lr=0.05
train_spp=64
ref_spp=512
iteration_count=200
seed0=12

ref_image=mi.render(scene,seed=seed0,spp=ref_spp)
mi.util.write_bitmap("gt_render.png", ref_image)

params = mi.traverse(scene)
if key_to_optimizer not in params:
    raise KeyError(f"Key '{key_to_optimizer}' not found in scene parameters.")
print("Before optimization:", params[key_to_optimizer])

losses=[]
psnrs=[]

para_ref=mi.Color3f(params[key_to_optimizer])
params[key_to_optimizer]=mi.Color3f(0.01,0.01,0.01)
ref_image=mi.render(scene,seed=seed0,spp=ref_spp)
mi.util.write_bitmap("gt_render2.png", ref_image)
opt=mi.ad.Adam(lr=optimizer_lr)
opt[key_to_optimizer]=params[key_to_optimizer]
params.update(opt)

for it in range(iteration_count):
    params.update()
    img=mi.render(scene,seed=seed0+it,spp=train_spp)
    loss=mse(img,ref_image)
    dr.backward(loss)
    opt.step()
    opt[key_to_optimizer]=dr.clamp(opt[key_to_optimizer],0,1)
   
    losses.append(to_scalar(loss))
    psnrs.append(to_scalar(psnr(img,ref_image)))
    if it%10==0 or it==iteration_count-1:
        print(f"it={it}, loss={losses[-1]:.6f}, psnr={psnrs[-1]:.2f}, {key_to_optimizer}={params[key_to_optimizer]}")
        mi.util.write_bitmap(f"iter_{it:04d}.png", img)
        mi.util.write_bitmap(f"diff_{it:04d}.png", dr.abs(img-ref_image)*10)
mi.util.write_bitmap("final_render.png", img)
print("After optimization:", params[key_to_optimizer])
print("Reference value:", para_ref)  
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Iterations')
plt.grid()
plt.savefig('loss_curve.png')
plt.clf()
plt.plot(psnrs)
plt.xlabel('Iteration')
plt.ylabel('PSNR')
plt.title('PSNR over Iterations')
plt.grid()
plt.savefig('psnr_curve.png')
plt.clf()
plt.close()
  