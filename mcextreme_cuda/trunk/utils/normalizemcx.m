function phi=normalizemcx(data,mua,Vvox,dt,energyabsorbed,energytotal)
alpha=energyabsorbed/energytotal/(Vvox*sum(data(:).*mua(:)))/(dt);
phi=alpha*data;
