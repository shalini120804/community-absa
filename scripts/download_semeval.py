from huggingface_hub import hf_hub_download
import shutil, os

REPO = "alexcadillon/SemEval2014Task4"
FILES = {
    "laptop_train.xml": "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml",
    "laptop_test.xml": "ABSA_Gold_TestData/Laptops_Test_Gold.xml",
    "rest_train.xml": "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml",
    "rest_test.xml": "ABSA_Gold_TestData/Restaurants_Test_Gold.xml",
}
os.makedirs("data/semeval", exist_ok=True)
for name, remote in FILES.items():
    p = hf_hub_download(REPO, remote, repo_type="dataset")
    shutil.copy(p, f"data/semeval/{name}")
    print("saved", name)