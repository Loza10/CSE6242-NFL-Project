from model.full_model.full_model import FullModel

if __name__ == "__main__":
    runner = FullModel(
        model_type=None,
        transformer_epochs=10,
        nn_epochs=15,
        lr=1e-3,
        num_particles=200,
        past_frames=10,
    )

    runner.run()