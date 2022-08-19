from txt2img import *
from pprint import pprint
import math

options = {
  "prompt":None,
  "outdir":"outputs/txt2img-samples",
  "skip_grid":False,
  "skip_save":False,
  "ddim_steps":70,
  "plms":False,
  "fixed_code":False,
  "ddim_eta":0.0,
  "n_iter":5,
  "H":512,
  "W":512,
  "C":4,
  "f":8,
  "n_samples":2,
  "n_rows":0,
  "scale":7.5,
  "config":"configs/stable-diffusion/v1-inference.yaml",
  "ckpt":"models/ldm/stable-diffusion-v1/model.ckpt",
  "seed":-1,
  "precision":"autocast"
}


def textPrompt(model):
  seed = options['seed']
  if seed == -1:
    seed = random.randint(1,4294967295)
  seed_everything(seed)
  
  config = OmegaConf.load(options['config'])

  if options['plms']:
      sampler = PLMSSampler(model)
  else:
      sampler = DDIMSampler(model)

  os.makedirs(options['outdir'], exist_ok=True)
  outpath = options['outdir']

  batch_size = options['n_samples']
  n_rows = options['n_rows']
  
  prompt = options['prompt']
  data = [batch_size * [prompt]]

  sample_path = os.path.join(outpath, "samples")
  os.makedirs(sample_path, exist_ok=True)
  base_count = len(os.listdir(sample_path))
  grid_count = len(os.listdir(outpath)) - 1
  
  n_rows = options['n_rows'] if options['n_rows'] > 0 else math.ceil(math.sqrt(options['n_iter']*options['n_samples']))
  
  start_code = None
  if options['fixed_code']:
    start_code = torch.randn([options['n_samples'], options['C'], options['H'] // options['f'], options['W'] // options['f']], device=device)

  precision_scope = autocast if options['precision']=="autocast" else nullcontext
  with torch.no_grad():
    with precision_scope("cuda"):
        with model.ema_scope():
          tic = time.time()
          all_samples = list()
          for n in trange(options['n_iter'], desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
              uc = None
              if options['scale'] != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
              if isinstance(prompts, tuple):
                prompts = list(prompts)
              c = model.get_learned_conditioning(prompts)
              shape = [options['C'], options['H'] // options['f'], options['W'] // options['f']]
              samples_ddim, _ = sampler.sample(S=options['ddim_steps'],
                                               conditioning=c,
                                               batch_size=options['n_samples'],
                                               shape=shape,
                                               verbose=False,
                                               unconditional_guidance_scale=options['scale'],
                                               unconditional_conditioning=uc,
                                               eta=options['ddim_eta'],
                                               x_T=start_code)

              x_samples_ddim = model.decode_first_stage(samples_ddim)
              x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

              if not options['skip_save']:
                for x_sample in x_samples_ddim:
                  x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                  Image.fromarray(x_sample.astype(np.uint8)).save(
                    os.path.join(sample_path, f"{base_count:05}.png"))
                  base_count += 1

              if not options['skip_grid']:
                all_samples.append(x_samples_ddim)

          if not options['skip_grid']:
            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1

          toc = time.time()

  print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
        f" \nEnjoy.")
          

def main():
  config = OmegaConf.load(options['config'])
  model = load_model_from_config(config, options['ckpt'])
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model = model.to(device)
  
  while True:
    print("Current options:")
    pprint(options)
    opt = input("1. Make a text prompt\n2. Make an image prompt\n3. Change an option\n4. Quit\n>")
    if opt == '1':
      while True:
        options['prompt'] = input("Prompt, or q to quit: ")
        if options['prompt'] == 'q':
          break
        textPrompt(model)
    elif opt == '2':
      print("Not implemented!")
    elif opt == '3':
      while True:
        change = input("Choose an option to modify, or q to return to the menu:")
        if change not in options.keys() and not change == 'q':
          continue
        elif change == 'q':
          break
        else:
          new = input(f"New value? Old was {options[change]}:")
          if new:
            try:
              options[change] = int(new)
            except ValueError:
              try:
                options[change] = float(new)
              except ValueError:
                options[change] = new
    else:
      break
  

if __name__ == "__main__":
    main()