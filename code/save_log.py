import wandb
import shutil
import os



PROJECT="mup_training"
ENTITY="mbzuai-llm"


        
def distill():
    #Currently logged in as: yusheng-su (mbzuai-llm). Use `wandb login --relogin` to force relogin
    #target_log = "../log/"+str(self.target_dir)+"/"+str(self.learning_rate)
    target_log = "../log/distill_loss"
    if os.path.isdir(target_log+"/wandb"):
        # delete dir
        shutil.rmtree(target_log+"/wandb")
    #create a new one
    os.makedirs(target_log+"/wandb")


    wandb.init(
        project=PROJECT,
        entity=ENTITY,
        #notes=socket.gethostname(),
        name="training_log",
        dir=target_log,
        job_type="training",
        reinit=True
    )

    #total_loss = 0
    # for i, batch in enumerate(prog):
    for i, j in enumerate(range(10)):
        # Your code here


        wandb.log(
            {
                "i": i,
                "j": j,
            },
            step=i,
        )

        '''
        wandb.log(
            {
                "loss": loss.item(),
            },
            step=i,
        )
        '''


def main():
    #distiller = Distiller()
    #distiller.distill()
    distill()

if __name__ == "__main__":
    main()