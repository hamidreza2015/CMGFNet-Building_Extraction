class ValidImageLoader(Dataset):

    def __init__(self,
                 image_paths,
                 target_paths,
                 dsm_path,
                 input_size = (224,224),
                 train=True):
      
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.dsm_path = dsm_path
        self.input_size = input_size

    def transform(self, image, mask, dsm):

      # Resize
      
      resize_im = transforms.Resize(size=self.input_size ,)
      image = resize_im(image)

      resize_lbl = transforms.Resize(size=self.input_size ,interpolation=Image.NEAREST,)
      mask = resize_lbl(mask)

      resize_dsm = transforms.Resize(size=self.input_size ,)
      dsm = resize_dsm(dsm)

      # Transform to tensor
      image = TF.to_tensor(image)
      dsm = TF.to_tensor(dsm)
      mask = torch.from_numpy(convert_from_color(np.array(mask)))

      # Normalized Data
      normalize_img = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
      

      image = normalize_img(image)
      
      normalize_dsm = transforms.Normalize(mean=[0.5], std=[0.5])
           
      dsm = normalize_dsm(dsm)


      return image, mask, dsm

    
    def __getitem__(self, index):

      image = Image.open(self.image_paths[index])
      mask = Image.open(self.target_paths[index])
      dsm = Image.open(self.dsm_path[index])
      
      x, y , z  = self.transform(image, mask, dsm)
      return x, y, z

    def __len__(self):
        return len(self.image_paths) 
