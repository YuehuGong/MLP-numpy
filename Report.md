- todo list
  - first try the whole pipeline with mlp[done]
    - implement mynn/lr_scheduler.py[done]
    - implement `MomentGD` in `optimizer.py`[done]
    - implement `runner.py`[done]
    - run test_train.py[done]
    - loss = inf;score is low:[done]
      - Crossloss:防止下降[done]
      - Xavier初始化[done]
    - Test score doesn't change:[done]
      - Cross backward
      - Linear forward
      - linear backward
  - implement moment mothod[done]
  - recheck cnn layers[done]
  - implement pooling layers[done]
  - implement cnnmodel[done]
  - test_train:
    - 判断太慢原因
  - Regulation:
    - earlystop[done]
    - dropout:[done]
    - L2


## Method
In this Project, I tried all methods mentioned in the introduction.

## Experiments

## Conclusion

## Result
- MLP+L2:0.95
- MLP+L2+dropout:0.9503
- MLP+L2+dropout+deeper&narrower structure:0.9617



