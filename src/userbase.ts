export class UserId {
    value: number;

    constructor() {
        this.value = 0;
    }
}

export class User {
    id: UserId;
    name: string;

    constructor(name: string = "Guest") {
        this.id = new UserId();
        this.name = name;
    }
}

export class UserRegistry {
    users: Array<User>;

    constructor() {
        this.users = [
            new User("John Doe"),
            new User("Jane Doe")
        ];
    }
}