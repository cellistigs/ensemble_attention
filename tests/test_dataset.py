from ensemble_attention.dataset import AdultDataset

def test_load_AdultDataset():
    path = "/Users/taigaabe/Downloads/"
    dataset = AdultDataset(path)

def test_get_AdultDataset():
    path = "/Users/taigaabe/Downloads/"
    dataset = AdultDataset(path)
    num,cat,target = dataset[[10]]
    print()
    assert list(num[0]) == [0.4914330734113408, 0.2225809502573476,-1.608987349074118,0.21667799017343095,-0.21562452708329816,-0.07914398664140074]
    assert list(cat[0]) == ['Private','10th','Married-civ-spouse','Craft-repair','Husband','White','Male','United-States']
    assert target[0] == False

def test_get_AdultDataset_quantile():
    path = "/Users/taigaabe/Downloads/"
    dataset = AdultDataset(path,transform = "quantile")
    num,cat,target = dataset[[10]]
    print()
    assert list(num[0]) == [0.5192253494916858, 0.45691381645085316,-1.529558186142137,1.4532795741807993,-5.199337582605575,-0.10305676609118142]
    assert list(cat[0]) == ['Private','10th','Married-civ-spouse','Craft-repair','Husband','White','Male','United-States']
    assert target[0] == False
