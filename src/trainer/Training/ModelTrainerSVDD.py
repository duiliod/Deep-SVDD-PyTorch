import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import seaborn as sns

class ModelTrainerSVDD(pl.LightningModule):
    def __init__(self, model, optimizer, loss, dataModule, monitor, activate_loss=None, center_eps=0.1, **config) -> None:
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._dataModule = dataModule
        self._monitor = monitor
        self._config = config
        self._activate_loss = activate_loss
        self._center_eps = center_eps   # coordinates of c closer to 0 than this are pushed to +-eps (0 disables it)


        # assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = 'one-class'

        # Deep SVDD parameters
        R = 0.0
        c = None
        self.nu = 0.1
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        
        # # Optimization parameters
        # self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated


    def on_save_checkpoint(self, checkpoint):
        # c is part of the model: without it the scores of a saved model cannot be reproduced
        checkpoint['svdd_center'] = None if self.c is None else self.c.detach().cpu()

    def on_load_checkpoint(self, checkpoint):
        c = checkpoint.get('svdd_center')
        if c is not None:
            self.c = c.to(self.device)

    def init_volume_stats(self, net):
        """Statistics of the structure volumes over the whole training set. They are stored in the network
        (buffers, so they travel with the checkpoint):
          - VolumeDifference: mean/std (voxels, shared by both hemispheres) of the volume input, a z-score
            instead of the old fixed (v-395)/4650.
          - VolumePassthrough: mean/std of (total volume, |L-R|/(L+R)) for the parameter-free features."""
        vols = []
        for inputs, _ in self._dataModule.train_dataloader():
            vols.append(torch.stack([inputs[1], inputs[2]], 1).double())
        vols = torch.cat(vols)
        if getattr(net, 'VolumeDifference', False):
            net.set_volume_stats(vols.mean().item(), vols.std().item())
            print('volume stats: mean %.1f voxels, std %.1f voxels, %d entries' % (vols.mean().item(), vols.std().item(), len(vols)))
        if getattr(net, 'VolumePassthrough', 0) > 0:
            f = torch.stack([vols.sum(1), (vols[:, 0] - vols[:, 1]).abs() / vols.sum(1)], 1)
            net.set_volume_feature_stats(f.mean(0), f.std(0))
            print('volume passthrough: alpha=%g, total volume %.1f +- %.1f voxels, asymmetry %.4f +- %.4f'
                  % (net.VolumePassthrough, f[:, 0].mean(), f[:, 0].std(), f[:, 1].mean(), f[:, 1].std()))

    def init_center_c_full(self, net, eps=None):
        """Center c = mean of the network outputs over the WHOLE training set (as in the original
        Deep SVDD). The previous version used the first batch of 12 samples only."""
        eps = self._center_eps if eps is None else eps
        if getattr(net, 'VolumeDifference', False) or getattr(net, 'VolumePassthrough', 0) > 0:
            self.init_volume_stats(net)
        n_out = net.rep_dim + (2 if getattr(net, 'VolumePassthrough', 0) > 0 else 0)
        c = torch.zeros(n_out, device=self.device)
        n_samples = 0
        was_training = net.training
        net.eval()
        with torch.no_grad():
            for inputs, _ in self._dataModule.train_dataloader():
                inputs = [inputs[0].to(self.device), inputs[1].to(self.device), inputs[2].to(self.device)]
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        net.train(was_training)
        c /= n_samples
        if eps > 0:
            # only the learned part of the output (the fixed volume features can not be matched with zero weights)
            cl = c[:net.rep_dim]
            cl[(abs(cl) < eps) & (cl < 0)] = -eps
            cl[(abs(cl) < eps) & (cl > 0)] = eps
        print('center c: mean |c|=%.4f, %d samples' % (c.abs().mean().item(), n_samples))
        return c

    def to(self, device):
        super().to(device)

    def configure_optimizers(self):
        '''
        Configuration of the opimizer used in the model
        '''
        cls = getattr(torch.optim, self._optimizer['Type'])
        optimizer = cls(self._model.parameters(), **{k:v for k,v in self._optimizer.items() if k!='Type'})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.2,
                                                               patience=15,
                                                               min_lr=1e-6,
                                                               verbose=True)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": 'mse_loss'
        },
    }

    def training_step(self, batch, batch_nb):
        '''
        training step inside the training loop
        it is made it for each batch of data
        '''
        x,y = batch
        #x=x.to(torch.float)
        y_hat = self._model.forward(x)
        # collapse monitors: out_norm -> 0 or out_std -> 0 means the network maps every sample to one point
        self.log('out_norm', y_hat.detach().norm(dim=1).mean(), on_step=False, on_epoch=True)
        self.log('out_std', y_hat.detach().std(dim=0).mean(), on_step=False, on_epoch=True)

        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c_full(self._model)
            print('Center c initialized.')

        scores = torch.sum((y_hat - self.c) ** 2, dim=1)
        loss = torch.mean(scores)

        monitor = self._monitor.onTrainingStep(self, self._model, x, y, y_hat, loss, {})
        return monitor

    def training_epoch_end(self, outputs):
        '''
        Function called at the end of the training loop in one epoch
        outputs: values saved in the return of the training_step
        '''
        self.log('mse_loss',torch.stack([x['loss'] for x in outputs]).mean())
        self._monitor.onTrainingEpochEnd(self, self._model, outputs, {})

    def validation_step(self, batch, batch_nb):
        '''
        Operates on a single batch of data from the validation set. In this step you'd might generate 
        examples or calculate anything of interest like Dice.
        '''
        idx_label_score = []
        x_val,y_val = batch
        y_hat_val = self._model.forward(x_val)
        scores_val = torch.sum((y_hat_val - self.c) ** 2, dim=1)

        idx_label_score += list(zip(y_val[0].cpu().data.numpy().tolist(),
                                    scores_val.cpu().data.numpy().tolist()))
        #x,y = batch
        #y_hat = self._model.forward(x.to(torch.float))
        #scores = torch.sum((y_hat - self.c) ** 2, dim=1)
        loss = torch.mean(scores_val)
               
        #idx_label_score += list(zip(y[0].cpu().data.numpy().tolist(),
        #                            scores.cpu().data.numpy().tolist()))
        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        monitor = self._monitor.onValidationStep(self, self._model, y_hat_val, labels, scores, loss, {})
        return monitor

    def validation_epoch_end(self, outputs):
        ''' al terminar la validacion se calcula el promedio de la loss de validacion
        '''
        salidas = torch.cat([x['yhat'] for x in outputs],dim=0)
        etiquetas = torch.cat([x['y'] for x in outputs],dim=0)
        salidas_tse = torch.cat([x['y_hat_val'] for x in outputs],dim=0)
        normales, anormales = [],[]

        for i in range(len(salidas)):
            if etiquetas[i]==0:
                normales.append(salidas[i])
            else:
                anormales.append(salidas[i])
        auc = roc_auc_score(etiquetas, salidas)

        self.log('d_normales',float(sum(normales) / len(normales)))
        self.log('d_anormales',float(sum(anormales) / len(anormales)))
        self.log('d_difference',float(sum(anormales) / len(anormales))-float(sum(normales) / len(normales)))
        self.log('AUC',auc)

        #if self.current_epoch%2==0:

         #   tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
          #  labels = np.array(etiquetas).reshape((139,1))
           # tensors =salidas_tse.cpu().detach().numpy()

            #labels = np.squeeze(labels)
            #tsne_results = tsne.fit_transform(tensors)

            #data = {}
            #data["x"] = tsne_results[:, 0]
            #data["y"] = tsne_results[:, 1]
            #data["z"] = labels

            #sns.scatterplot(x="x", y="y", hue="z", data=data, palette="viridis", legend="full", s=10)
            #plt.savefig('../results/experiment_2b/tsne'+str(self.current_epoch)+'.png')
            #plt.close()

        samples = self._dataModule.val_sample(3)
        monitor = self._monitor.onValidationEpochEnd(self, self._model, samples, outputs, {})
        return monitor

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Operates on a single batch of data from the test set. 
        In this step you’d normally generate examples or calculate anything of interest such as accuracy.
        '''
        #x,y = batch
        #y_hat = self._model.forward(x)
        scores = torch.sum((y_hat - self.c) ** 2, dim=1)
        loss = torch.mean(scores)

        monitor = self._monitor.onTrainingStep(self, self._model, x, y, y_hat, loss, {})
        return monitor

    def test_epoch_end(self, outputs):
        monitor = self._monitor.onTestEpochEnd(self, self._model, outputs, {})
        return monitor

    def forward(self, batch):
        '''
        To predict the output of the model
        '''
        x,y = batch
        return self._model.predict(x)


    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
        """
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            inputs, _= train_loader
            #inputs = inputs.to(self.device)
            inputs = [inputs[0].to(self.device),inputs[1].to(self.device),inputs[2].to(self.device)]  #$$$#
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
        net.train()
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


