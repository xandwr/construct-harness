export class Group {
    id: string;

    constructor(id: string) {
        this.id = id;
    }
}

export class User {
    id: number;
    name: string;
    groups: Array<Group>;

    constructor(name: string = "Guest", groups: Array<Group> = []) {
        this.id = 0;
        this.name = name;
        this.groups = groups;
    }
}

export class UserRegistry {
    users: Array<User>;

    constructor() {
        this.users = [
            new User("John Doe", [new Group("wheel"), new Group("video")]),
            new User("Jane Doe")
        ];
    }
}