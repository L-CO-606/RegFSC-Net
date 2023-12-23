# RegFSC-Net: Medical Image Registration via Fourier Transform with Spatial Reorganization and Channel Refinement Network

**Keywords：Deformable image registration, Medical image registration, Unsupervised registration, Brain MRI**


## Training and Inference
Train and Test
```python
python FSC_train.py --using_l2 2 --smth_labda 5.0 --lr 1e-4
```
```python
python FSC_infer.py --using_l2 2 --smth_labda 5.0 --lr 1e-4
```
## Implementation of the Variants
If you want to try the diffeomorphic method, please execute RegFSC-Net-Diff:
```python
python FSC_Diff_train.py --using_l2 2 --smth_labda 1.0 --lr 1e-4
```
```python
python FSC_Diff_infer.py --using_l2 2 --smth_labda 1.0 --lr 1e-4
```
If you want to perform registration in a low VRAM environment, please try RegFSC-Net-Small:
```python
python FSC_S_train.py --using_l2 2 --smth_labda 1.0 --lr 1e-4
```
```python
python FSC_S_infer.py --using_l2 2 --smth_labda 1.0 --lr 1e-4
```
If you have sufficient VRAM resources, please try RegFSC-Net-Large for achieving the highest registration accuracy: 
```python
python FSC_L_train.py --using_l2 2 --smth_labda 5.0 --lr 1e-4
```
```python
python FSC_L_infer.py --using_l2 2 --smth_labda 5.0 --lr 1e-4
```
## Reference
<a href="https://github.com/voxelmorph/voxelmorph">VoxelMorph</a>,
<a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration">TransMorph</a>,
<a href="https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks">SYM-Net</a>,
<a href="https://github.com/xi-jia/Fourier-Net">Fourier-Net</a>
and
<a href="https://github.com/cwmok/LapIRN">LapIRN</a>.

