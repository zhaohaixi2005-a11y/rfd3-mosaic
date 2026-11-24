# Ho-Yeung LHD101 C3 initializer

This branch ports the starting-layout algorithm from the [public Ho-Yeung
interface-seed source](https://github.com/Khmelinskaia-Lab/RFdiffusion_interfaceseed/blob/e429babbd32ba64394eb1039a3ff952c0f0560c9/rfdiffusion/inference/model_runners.py#L311-L445).
The BSD notice is in [HO_YEUNG_INITIALIZER_BSD_NOTICE.txt](HO_YEUNG_INITIALIZER_BSD_NOTICE.txt).
It is scoped to the published LHD101 C3 interface and two exact input-file
encodings. The public source revision is not the historical `1479db7` variant.

```sh
rfd3-mosaic prepare-poses examples/rfd3_mosaic/simple_interface_seed.yaml --output-dir ./lhd101-poses --count 10 --seed 0
rfd3-mosaic run ./lhd101-poses/trial-0000/mosaic.yaml
```

Run the other emitted YAMLs the same way on this public base. Each design gets
one fresh draw: integer Euler angles, an x translation, one of two C3 neighbors,
and one integer linker length reused across its three copies. No candidate pool,
rank, top-k, deduplication or failed-attempt replacement is used. The ledger
preserves individual draw IDs and failures. Pose RNG is independent of Mosaic's
diffusion RNG. This is an initialization change, not a replacement of the
RFdiffusion3 diffusion process or output gates. It does not promise conformational
quality, successful generation, or parity with the original RFdiffusion model.
