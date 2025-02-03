from invoke import task
import os
@task
def git(ctx, message):
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")
