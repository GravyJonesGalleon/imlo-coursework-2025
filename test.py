import assessment_utils as au

if __name__ == "__main__":
    try:
        model = au.load_model(au.model_save_path)
    except FileNotFoundError:
        print(f"No file exists at {au.model_save_path}")
        quit(1)

    au.test(au.test_dataloader, model, au.loss_fn)
