train = load('optdigits_train.txt');
valid = load('optdigits_valid.txt');
test = load('optdigits_test.txt');
m = [3 6 9 12 15 18];

E1 = [];
E2 = [];
E3 = [];
for i = 1 : length(m)
    [z w v e1 e2] = mlptrain(train,valid,m(i),10);
    [z2 e3] = mlptest(test,w,v);
    E1 = [E1 e1];
    E2 = [E2 e2];
    E3 = [E3 e3];
    sprintf('finished %d',i)
end

hold all
plot(m,E1);
plot(m,E2);
plot(m,E3);
xlabel('number of m');
ylabel('train/valid/test error rates');
title('comparing 3 error rates with each m value');