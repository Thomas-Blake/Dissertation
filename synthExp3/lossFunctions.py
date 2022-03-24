import torch

def CEL(pred, y):
  return torch.mean(torch.log(torch.sum(torch.exp(pred),1))-torch.sum(torch.mul(pred,y),1))

def adapativeLoss(pred, y , priors):
  #edy = torch.exp(torch.pow(priors,-0.25))
  edy = torch.sum(torch.mul(torch.exp(torch.pow(priors,-0.25)),y),1)
  #print(edy.shape)
  edy = torch.transpose(edy.repeat(11,1),0,1)
  #print(edy)
  

  fy = torch.sum(torch.mul(pred,y),1)
  fy = torch.transpose(fy.repeat(11,1),0,1)
  term1 = torch.exp(pred-fy)
  term2 = torch.sum(torch.mul(edy,term1),1)-torch.sum(torch.mul(edy,y),1)
  term3 = torch.log(1+term2)
  return torch.sum(term3)

def equalisedLoss(pred, y , priors):
  edy = torch.exp(torch.log(priors))
  fy = torch.sum(torch.mul(pred,y),1)
  fy = torch.transpose(fy.repeat(11,1),0,1)
  term1 = torch.exp(pred-fy)
  term2 = torch.sum(torch.mul(edy,term1),1)-torch.sum(torch.mul(edy,y),1)
  term3 = torch.log(1+term2)
  return torch.sum(term3)

def logitAdjusted(pred,y,priors):
  edy1 = torch.sum(torch.mul(priors,y),1)
  edy1 = torch.transpose(edy1.repeat(11,1),0,1)
  edy2 = torch.exp(torch.log(priors)-torch.log(edy1))
  fy = torch.sum(torch.mul(pred,y),1)
  fy = torch.transpose(fy.repeat(11,1),0,1)
  term1 = torch.exp(pred-fy)
  term2 = torch.sum(torch.mul(edy2,term1),1)-torch.sum(torch.mul(edy2,y),1)
  term3 = torch.log(1+term2)
  return torch.sum(term3)

