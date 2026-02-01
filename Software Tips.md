Creating a tar copy of my repo without large files: `tar -czf /workspaces/neuralODEs/neuralODEs_backup_2026-01-26.tar.gz '--exclude=.git' '--exclude=**/wandb/*' '--exclude=*.eqx' '--exclude=neuralODEs_backup_2026-01-26.tar.gz' -C /workspaces/neuralODEs .`

```
cd /workspaces/neuralODEs
git ls-files -z --cached --modified --others --exclude-standard | \
  tar --null -T - -czf /workspaces/neuralODEs/neuralODEs_backup_2026-02-01.tar.gz \
  --exclude='**/wandb/*' --exclude='**/artifacts/*' --exclude='*.eqx'
```