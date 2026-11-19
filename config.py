from dataclasses import dataclass
from pathlib import Path

EXCLUDED = frozenset({'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', 'node_modules', '.tox', 'dist', 'build', 'coverage', '.next', 'backups', 'logs'})

@dataclass(frozen=True)
class Config:
    root: Path
    backup_root: Path
    allowed_paths: tuple[Path, ...] = ()
    read_only_paths: tuple[Path, ...] = ()

    @classmethod
    def create(cls, root: str, backup_root: str | None = None, *, allowed_paths: str | None = None, read_only_paths: str | None = None):
        path = Path(root).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError('Project path must be a directory')
        backups = Path(backup_root).expanduser().resolve() if backup_root else (Path(__file__).resolve().parent / 'backups').resolve()
        if backups == path or path in backups.parents:
            raise ValueError('Backups must be outside the target project')
        if backups == path.parent and path.name == 'backups':
            raise ValueError('Invalid backup location')
        backups.mkdir(parents=True, exist_ok=True)
        def parse_scopes(raw: str | None, option: str) -> tuple[Path, ...]:
            if not raw:
                return ()
            scopes = []
            for value in raw.split(','):
                value = value.strip()
                if not value:
                    raise ValueError(f'{option}: empty path is not allowed')
                rel = Path(value)
                if rel.is_absolute() or '..' in rel.parts or rel == Path('.'):
                    raise ValueError(f'{option}: paths must be relative and cannot contain .. or .')
                if any(part in EXCLUDED for part in rel.parts):
                    raise ValueError(f'{option}: protected directory')
                resolved = (path / rel).resolve(strict=False)
                if not resolved.is_relative_to(path):
                    raise ValueError(f'{option}: path escapes repository')
                scopes.append(resolved)
            return tuple(scopes)

        return cls(path, backups, parse_scopes(allowed_paths, '--allowed-paths'),
                   parse_scopes(read_only_paths, '--read-only-paths'))

    @staticmethod
    def _matches(target: Path, scopes: tuple[Path, ...]) -> bool:
        return any(target == scope or scope in target.parents for scope in scopes)

    def check_edit(self, target: Path) -> None:
        if target.suffix.lower() != '.py':
            raise PermissionError('Only Python .py files may be edited')
        # Deny rules take precedence over allow rules, including nested paths.
        if self._matches(target, self.read_only_paths):
            raise PermissionError(f'Read-only path: {target.relative_to(self.root)}')
        if self.allowed_paths and not self._matches(target, self.allowed_paths):
            raise PermissionError(f'Editing outside --allowed-paths: {target.relative_to(self.root)}')

    def file(self, name: str, *, must_exist: bool = True, for_edit: bool = False) -> Path:
        candidate = Path(name)
        if candidate.is_absolute() or '..' in candidate.parts or not candidate.parts:
            raise ValueError('Use a project-relative path without ..')
        if any(part in EXCLUDED for part in candidate.parts):
            raise ValueError('Protected directory')
        target = (self.root / candidate).resolve(strict=False)
        if not target.is_relative_to(self.root):
            raise ValueError('Path escapes project (possibly through symlink)')
        forbidden = {'.env', '.npmrc', '.pypirc', 'id_rsa', 'id_ed25519'}
        binary_ext = {'.png','.jpg','.jpeg','.gif','.webp','.pdf','.zip','.gz','.tar','.7z','.exe','.dll','.so','.dylib','.class','.pyc','.woff','.woff2','.ttf','.mp3','.mp4','.sqlite','.db','.pem','.key','.p12','.pfx','.lock'}
        if target.name.lower() in forbidden or target.name.lower().startswith('.env.') or target.suffix.lower() in binary_ext:
            raise ValueError('Protected or binary file')
        if target.is_file() and b'\x00' in target.read_bytes()[:8192]:
            raise ValueError('Binary file not supported')
        if target.suffix.lower() != '.py':
            raise ValueError('Only Python .py files are supported')
        if for_edit:
            self.check_edit(target)
        if must_exist and not target.is_file():
            raise FileNotFoundError(name)
        return target
